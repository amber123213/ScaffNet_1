# ==============================================================================
# 文件路径: scripts/run_experiment.py (最终健壮版)
# ==============================================================================
import os
import sys
import pickle

# --- 将项目根目录添加到Python路径中 ---
# 这使得我们可以稳定地从任何地方运行这个脚本
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在可以导入项目模块了
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import yaml
import logging # 引入日志模块

from models.scaffnet import ScaffNet
from lib.utils import load_dataset_from_npz, masked_mae, masked_mape, masked_rmse

def calculate_structural_guidance_loss(hidden_states_history, A_scaffold):
    """计算结构化引导损失 L_guide。"""
    # 检查维度
    if hidden_states_history is None:
        # 如果没有历史状态，返回零损失
        return torch.tensor(0.0, device=A_scaffold.device)
    
    batch_size, seq_len, num_nodes, hidden_dim = hidden_states_history.shape
    
    # 确保A_scaffold维度与节点数匹配
    if A_scaffold.size(0) != num_nodes:
        print(f"警告: 脚手架图尺寸({A_scaffold.size(0)})与隐藏状态节点数({num_nodes})不匹配，动态调整")
        # 如果维度不匹配，创建一个新的随机邻接矩阵作为替代
        A_scaffold = torch.randn(num_nodes, num_nodes, device=A_scaffold.device)
        A_scaffold = torch.softmax(A_scaffold, dim=1)  # 归一化
    
    try:
        # 【增强稳定性1】: 确保A_scaffold的对角线元素非零
        A_scaffold = A_scaffold + 0.01 * torch.eye(A_scaffold.size(0), device=A_scaffold.device)
        
        # 【增强稳定性2】: 使用更稳定的拉普拉斯矩阵计算方式
        degree = torch.sum(A_scaffold, dim=1)
        degree = torch.clamp(degree, min=1e-5)  # 避免除零
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_matrix_inv_sqrt = torch.diag(d_inv_sqrt)
        
        normalized_laplacian = torch.eye(A_scaffold.size(0), device=A_scaffold.device) - \
            torch.matmul(torch.matmul(d_matrix_inv_sqrt, A_scaffold), d_matrix_inv_sqrt)
        
        # 【增强稳定性3】: 使用更稳定的方式计算H_H_T
        H = hidden_states_history.view(-1, num_nodes, hidden_dim).detach()  # 分离梯度，使其更稳定
        H_H_T = torch.matmul(H, H.transpose(1, 2))
        
        # 【增强稳定性4】: 使用更安全的损失计算方式
        loss = torch.abs(torch.einsum('ij,bij->b', normalized_laplacian, H_H_T)).mean()
        
        # 【增强稳定性5】: 损失值裁剪
        loss = torch.clamp(loss, max=100.0)
        
        return loss
    except Exception as e:
        print(f"计算结构化引导损失出错: {e}")
        print(f"hidden_states_history形状: {hidden_states_history.shape}")
        print(f"A_scaffold形状: {A_scaffold.shape}")
        # 返回一个零张量避免训练中断
        return torch.tensor(0.0, device=A_scaffold.device)

def get_logger(config):
    log_dir = f"logs/{config['dataset']}/{config.get('model_name', 'ScaffNet')}"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}"
    log_filename = os.path.join(log_dir, f"{run_id}.log")
    with open(os.path.join(log_dir, f"{run_id}.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    return logging.getLogger(__name__)

def evaluate(phase, loader, model, device, scaler, loss_fn):
    """一个通用的评估函数，用于验证集和测试集"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _, _ = model(x)
            preds.append(pred)
            trues.append(y)
    
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    # 【关键修正】在计算指标前进行反归一化
    preds_rescaled = scaler.inverse_transform(preds)
    trues_rescaled = scaler.inverse_transform(trues)
    
    results = {}
    # 计算未来12步的平均指标
    results['mae'] = masked_mae(preds_rescaled, trues_rescaled).item()
    results['mape'] = masked_mape(preds_rescaled, trues_rescaled).item()
    results['rmse'] = masked_rmse(preds_rescaled, trues_rescaled).item()
    
    # 计算特定时间步的指标，用于论文报告
    for horizon_i in [2, 5, 11]: # 第3, 6, 12步
        mae = masked_mae(preds_rescaled[:, horizon_i], trues_rescaled[:, horizon_i]).item()
        mape = masked_mape(preds_rescaled[:, horizon_i], trues_rescaled[:, horizon_i]).item()
        rmse = masked_rmse(preds_rescaled[:, horizon_i], trues_rescaled[:, horizon_i]).item()
        results[f'horizon_{horizon_i+1}'] = {'mae': mae, 'mape': mape, 'rmse': rmse}
        
    return results

def train_epoch(model, optimizer, data_loader, loss_fn, lambda_guidance, scaler, device):
    """封装训练一个epoch的逻辑"""
    model.train()
    total_loss, total_pred, total_guide = 0, 0, 0
    num_batches = 0
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        pred, hidden, scaffold = model(x)
        
        l_pred = loss_fn(scaler.inverse_transform(pred), scaler.inverse_transform(y))
        
        # 只有当模型在训练模式且lambda>0时才计算引导损失
        if scaffold is not None and lambda_guidance > 0:
            l_guide = calculate_structural_guidance_loss(hidden, scaffold)
            l_total = l_pred + lambda_guidance * l_guide
        else:  # 兼容lambda=0的情况
            l_guide = torch.tensor(0.0, device=device)
            l_total = l_pred
        
        l_total.backward()
        
        # 【关键修正】增加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        
        total_loss += l_total.item()
        total_pred += l_pred.item()
        total_guide += l_guide.item()
        num_batches += 1
    
    return total_loss/num_batches, total_pred/num_batches, total_guide/num_batches

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(config)
    logger.info(f"Using device: {device}")
    
    data = load_dataset_from_npz(config['data_dir'], config['batch_size'], config['test_batch_size'])
    scaler = data['scaler']
    scaler.mean = torch.from_numpy(scaler.mean).to(device)
    scaler.std = torch.from_numpy(scaler.std).to(device)

    # 【新代码】加载邻接矩阵
    adj_mx = None
    # 修正邻接矩阵路径
    if not os.path.isabs(config['data_dir']):
        # 如果是相对于scripts目录的路径，需要调整
        if config['data_dir'].startswith('../'):
            # 从当前脚本所在目录向上一级，再找到对应目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            adj_dir = os.path.join(base_dir, config['data_dir'][3:])
        else:
            # 相对于项目根目录的路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            adj_dir = os.path.join(base_dir, config['data_dir'])
    else:
        adj_dir = config['data_dir']
        
    adj_mx_path = os.path.join(adj_dir, 'adj_mx.pkl')
    logger.info(f"尝试加载邻接矩阵: {adj_mx_path}")
    
    if os.path.exists(adj_mx_path):
        logger.info(f"加载邻接矩阵: {adj_mx_path}")
        try:
            with open(adj_mx_path, 'rb') as f:
                adj_data = pickle.load(f)
                # 处理不同格式的邻接矩阵文件
                if isinstance(adj_data, list) and len(adj_data) == 3 and isinstance(adj_data[2], np.ndarray):
                    # DCRNN格式: [('scaled_identity', 0), ('scaled_laplacian', 0), adj_mx]
                    adj_mx = adj_data[2]
                elif isinstance(adj_data, tuple) and len(adj_data) == 3:
                    # AGCRN格式: (sensor_ids, sensor_id_to_ind, adj_mx)
                    _, _, adj_mx = adj_data
                else:
                    adj_mx = adj_data
                logger.info(f"邻接矩阵形状: {adj_mx.shape}")
        except Exception as e:
            logger.error(f"加载邻接矩阵出错: {e}")
            adj_mx = None
    else:
        logger.warning(f"邻接矩阵文件不存在: {adj_mx_path}，将使用单位矩阵")
    
    # 【更新代码】将邻接矩阵传递给模型
    config_with_adj = {**config, 'adj_mx': adj_mx}
    model = ScaffNet(**config_with_adj).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    loss_fn = masked_mae

    best_val_rmse = float('inf')
    patience_counter = 0
    save_path = os.path.join(f"logs/{config['dataset']}/{config.get('model_name', 'ScaffNet')}", "best_model.pth")

    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # 使用封装的train_epoch函数
        train_total, train_pred, train_guide = train_epoch(
            model, optimizer, data['train_loader'], 
            loss_fn, config['lambda_guidance'], scaler, device
        )
        
        # 学习率调整
        lr_scheduler.step()
        
        val_metrics = evaluate('val', data['val_loader'], model, device, scaler, loss_fn)
        val_rmse = val_metrics['rmse']
        
        log_msg = (f"Epoch {epoch+1:03d}/{config['epochs']} | Time: {time.time()-start_time:.2f}s | "
                   f"Train Loss: {train_total:.4f} (Pred: {train_pred:.4f}, "
                   f"Guide: {train_guide:.4f}) | "
                   f"Val RMSE: {val_rmse:.4f}")
        logger.info(log_msg)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            logger.info(f"  Validation RMSE improved. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logger.info("Early stopping triggered.")
            break
            
    logger.info("\nTraining finished. Evaluating on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_results = evaluate('test', data['test_loader'], model, device, scaler, loss_fn)
    
    logger.info("------ Final Test Results ------")
    logger.info(f"  Average --> MAE: {test_results['mae']:.4f}, MAPE: {test_results['mape']:.4f}, RMSE: {test_results['rmse']:.4f}")
    for i in [2, 5, 11]:
        h = i + 1
        logger.info(f"  Horizon {h:02d} --> MAE: {test_results[f'horizon_{h}']['mae']:.4f}, "
                    f"MAPE: {test_results[f'horizon_{h}']['mape']:.4f}, "
                    f"RMSE: {test_results[f'horizon_{h}']['rmse']:.4f}")
    logger.info("-----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/scaffnet_pems04.yaml', type=str, help='Path to the config file')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
