import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import argparse
import yaml

# 将项目根目录添加到Python路径中，以便import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
import model.scaffnet as models # 动态加载模型
from data_loader.traffic_state_dataset import TrafficStateDataset
from lib.utils import masked_mae, masked_mape, masked_rmse, get_scaler

# ------------------------------------------------------------------------------
# 损失函数与评估函数
# ------------------------------------------------------------------------------
def calculate_structural_guidance_loss(hidden_states_history, A_scaffold):
    """计算结构化引导损失 L_guide。"""
    degree = torch.sum(A_scaffold, dim=1)
    d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
    d_matrix_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = torch.eye(A_scaffold.size(0), device=A_scaffold.device) - \
                           torch.matmul(torch.matmul(d_matrix_inv_sqrt, A_scaffold), d_matrix_inv_sqrt)
    batch_size, seq_len, num_nodes, hidden_dim = hidden_states_history.shape
    H = hidden_states_history.view(-1, num_nodes, hidden_dim)
    H_H_T = torch.matmul(H, H.transpose(1, 2))
    loss = torch.einsum('ij,bij->b', normalized_laplacian, H_H_T).sum()
    return loss / (batch_size * seq_len)

def train_epoch(model, data_loader, optimizer, pred_loss_fn, lambda_guidance, device, scaler):
    model.train()
    total_loss, total_pred_loss, total_guide_loss = 0, 0, 0
    for batch in data_loader:
        batch.to_torch(device)
        optimizer.zero_grad()
        y_pred, hidden_states, A_scaffold = model(batch)
        y_true = batch['y']
        l_pred = pred_loss_fn(scaler.inverse_transform(y_pred), scaler.inverse_transform(y_true))
        l_guide = calculate_structural_guidance_loss(hidden_states, A_scaffold)
        l_total = l_pred + lambda_guidance * l_guide
        l_total.backward()
        optimizer.step()
        total_loss += l_total.item()
        total_pred_loss += l_pred.item()
        total_guide_loss += l_guide.item()
    return total_loss/len(data_loader), total_pred_loss/len(data_loader), total_guide_loss/len(data_loader)

def evaluate_epoch(model, data_loader, device, scaler):
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch.to_torch(device)
            y_pred, _, _ = model(batch)
            y_preds.append(y_pred)
            y_trues.append(batch['y'])
    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    y_preds_rescaled = scaler.inverse_transform(y_preds)
    y_trues_rescaled = scaler.inverse_transform(y_trues)
    return {
        'mae': masked_mae(y_preds_rescaled, y_trues_rescaled).item(),
        'mape': masked_mape(y_preds_rescaled, y_trues_rescaled).item(),
        'rmse': masked_rmse(y_preds_rescaled, y_trues_rescaled).item()
    }

# ------------------------------------------------------------------------------
# 主执行逻辑
# ------------------------------------------------------------------------------
def main(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("加载数据...")
    dataset_class = getattr(sys.modules[__name__], config['dataset_class'])
    train_dataset = dataset_class(config['dataset_config'])
    val_dataset = dataset_class(config['dataset_config'], subset='val')
    test_dataset = dataset_class(config['dataset_config'], subset='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)
    
    scaler = get_scaler(train_dataset)

    print("初始化模型...")
    model_class = getattr(models, config['model_class'])
    model_config = config['model_config']
    model_config['num_nodes'] = train_dataset.num_nodes
    model = model_class(**model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config['train']['lr_scheduler_args'])
    
    prediction_loss_fn = masked_mae
    
    print(f"开始训练 {config['model_class']} on {config['dataset_config']['dataset']}...")
    best_val_rmse = float('inf')
    patience_counter = 0
    save_dir = f"saved_models/{config['dataset_config']['dataset']}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(config['train']['epochs']):
        start_time = time.time()
        
        train_total, train_pred, train_guide = train_epoch(
            model, train_loader, optimizer, prediction_loss_fn, config['train']['lambda_guidance'], device, scaler
        )
        val_metrics = evaluate_epoch(model, val_loader, device, scaler)
        
        lr_scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:03d}/{config['train']['epochs']} | Time: {epoch_time:.2f}s | Train Loss: {train_total:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.4f}, Val MAPE: {val_metrics['mape']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            patience_counter = 0
            print(f"  Validation RMSE improved. Saving model...")
            torch.save(model.state_dict(), f"{save_dir}/{config['model_class']}_best.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config['train']['patience']:
            print("Early stopping triggered.")
            break

    print("\n训练完成。在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(f"{save_dir}/{config['model_class']}_best.pth"))
    test_metrics = evaluate_epoch(model, test_loader, device, scaler)
    print("------ 测试结果 ------")
    for metric, value in test_metrics.items():
        print(f"  Test {metric.upper()}: {value:.4f}")
    print("----------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/scaffnet_pemsd4.yaml', type=str,
                        help='Configuration file path.')
    args = parser.parse_args()
    main(args) 