import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import argparse
import yaml

# 将项目根目录添加到Python路径中，以便import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.scaffnet import ScaffNet
from lib.utils import load_dataset, masked_mae, masked_mape, masked_rmse

def calculate_structural_guidance_loss(hidden_states_history, A_scaffold):
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

def run(config, model, data_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    scaler = data_loader['scaler']
    loss_fn = masked_mae

    print(f"开始训练 {model.__class__.__name__}...")
    best_val_rmse = float('inf')
    patience_counter = 0
    save_dir = os.path.join(project_root, f"saved_models/{config['base_dir'].split('/')[-1]}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    best_model_path = f"{save_dir}/{model.__class__.__name__}_best.pth"

    for epoch in range(config['epochs']):
        # --- 训练 ---
        model.train()
        start_time = time.time()
        for x, y in data_loader['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, hidden, scaffold = model(x)
            l_pred = loss_fn(scaler.inverse_transform(pred), scaler.inverse_transform(y))
            l_guide = calculate_structural_guidance_loss(hidden, scaffold)
            l_total = l_pred + config['lambda_guidance'] * l_guide
            l_total.backward()
            optimizer.step()
        
        # --- 验证 ---
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in data_loader['val_loader']:
                x, y = x.to(device), y.to(device)
                pred, _, _ = model(x)
                val_preds.append(pred)
                val_trues.append(y)
        val_preds = torch.cat(val_preds, dim=0)
        val_trues = torch.cat(val_trues, dim=0)
        val_rmse = masked_rmse(scaler.inverse_transform(val_preds), scaler.inverse_transform(val_trues)).item()

        print(f"Epoch {epoch+1:03d} | Time: {time.time()-start_time:.2f}s | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  Validation RMSE improved. Model saved.")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break
        lr_scheduler.step()
            
    # --- 测试 ---
    print("\n训练完成。在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(best_model_path))
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in data_loader['test_loader']:
            x, y = x.to(device), y.to(device)
            pred, _, _ = model(x)
            test_preds.append(pred)
            test_trues.append(y)
    test_preds = torch.cat(test_preds, dim=0)
    test_trues = torch.cat(test_trues, dim=0)
    test_preds_rescaled = scaler.inverse_transform(test_preds)
    test_trues_rescaled = scaler.inverse_transform(test_trues)

    print("------ 测试结果 ------")
    for horizon_i in [2, 5, 11]: # 对应第3, 6, 12个预测步
        mae = masked_mae(test_preds_rescaled[..., horizon_i, :], test_trues_rescaled[..., horizon_i, :]).item()
        mape = masked_mape(test_preds_rescaled[..., horizon_i, :], test_trues_rescaled[..., horizon_i, :]).item()
        rmse = masked_rmse(test_preds_rescaled[..., horizon_i, :], test_trues_rescaled[..., horizon_i, :]).item()
        print(f"Horizon {horizon_i+1:02d} -> MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
    print("----------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/scaffnet_pems04.yaml', type=str, help='Config file path')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 确保数据路径是相对于项目根目录的
    dataset_dir = os.path.join(project_root, config['base_dir'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(dataset_dir, config['batch_size'], config['test_batch_size'])

    model = ScaffNet(**config).to(device)

    run(config, model, data, device) 