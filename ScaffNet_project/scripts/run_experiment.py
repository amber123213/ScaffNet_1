# 文件路径: scripts/run_experiment.py
import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import argparse
import yaml

# 将项目根目录添加到Python路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scaffnet import ScaffNet
from lib.utils import load_dataset_from_npz, masked_mae, masked_mape, masked_rmse

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

def run_experiment(model, data, optimizer, lr_scheduler, loss_fn, lambda_guidance, scaler, device, config):
    best_val_rmse = float('inf')
    patience_counter = 0
    save_dir = f"saved_models/{config['dataset']}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    print(f"开始训练 {config['model_name']} on {config['dataset']}...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # --- 训练 ---
        model.train()
        total_loss, total_pred, total_guide = 0, 0, 0
        for x, y in data['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, hidden, scaffold = model(x)
            
            # 使用带掩码的损失函数
            l_pred = loss_fn(scaler.inverse_transform(pred), scaler.inverse_transform(y))
            l_guide = calculate_structural_guidance_loss(hidden, scaffold)
            l_total = l_pred + lambda_guidance * l_guide
            
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # 梯度裁剪
            optimizer.step()
            
            total_loss += l_total.item()
            total_pred += l_pred.item()
            total_guide += l_guide.item()
        
        lr_scheduler.step()
        
        # --- 验证 ---
        model.eval()
        val_preds = []
        with torch.no_grad():
            for x, y in data['val_loader']:
                x, y = x.to(device), y.to(device)
                pred, _, _ = model(x)
                val_preds.append(pred)
        
        val_preds = torch.cat(val_preds, dim=0)
        val_rmse = masked_rmse(scaler.inverse_transform(val_preds), data['y_val']).item()
        
        print(f"Epoch {epoch+1:03d}/{config['epochs']} | Time: {time.time()-start_time:.2f}s | Train Loss: {total_loss/len(data['train_loader']):.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/{config['model_name']}_best.pth")
            print(f"  Validation RMSE improved. Model saved to {save_dir}/{config['model_name']}_best.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break
            
    # --- 测试 ---
    print("\nTraining finished. Evaluating on test set...")
    model.load_state_dict(torch.load(f"{save_dir}/{config['model_name']}_best.pth"))
    test_preds = []
    with torch.no_grad():
        for x, y in data['test_loader']:
            x, y = x.to(device), y.to(device)
            pred, _, _ = model(x)
            test_preds.append(pred)
            
    test_preds = torch.cat(test_preds, dim=0)
    test_preds_rescaled = scaler.inverse_transform(test_preds)
    test_trues_rescaled = data['y_test_raw']

    print("------ Test Results (Horizon 3, 6, 12) ------")
    for horizon_i in [2, 5, 11]: 
        mae = masked_mae(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        mape = masked_mape(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        rmse = masked_rmse(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        print(f"Horizon {horizon_i+1:02d} -> MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/scaffnet_pems04.yaml', type=str, help='Config file path')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_dataset_from_npz(config['data_dir'], config['batch_size'], config['test_batch_size'])
    scaler = data['scaler']
    scaler.mean = torch.from_numpy(scaler.mean).to(device)
    scaler.std = torch.from_numpy(scaler.std).to(device)

    model_config = {k: v for k, v in config.items() if k not in ['data_dir', 'batch_size', 'test_batch_size', 'epochs', 'patience', 'learning_rate', 'lambda_guidance', 'lr_decay_rate', 'lr_decay_step']}
    model_config['model_name'] = 'ScaffNet'
    
    model = ScaffNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    
    run_experiment(model, data, optimizer, lr_scheduler, masked_mae, config['lambda_guidance'], scaler, device, config)
