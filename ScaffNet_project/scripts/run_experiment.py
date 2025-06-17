import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import argparse
import yaml

sys.path.append(os.path.join(os.getcwd(), "..")) # 将项目根目录添加到路径
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

def run(model, data, optimizer, lr_scheduler, loss_fn, lambda_guidance, scaler, device, config):
    best_val_rmse = float('inf')
    patience_counter = 0
    save_dir = f"saved_models/{config['base_dir'].split('/')[-1]}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for epoch in range(config['epochs']):
        # --- 训练 ---
        model.train()
        total_loss, total_pred, total_guide = 0, 0, 0
        for x, y in data['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, hidden, scaffold = model(x)
            l_pred = loss_fn(scaler.inverse_transform(pred), scaler.inverse_transform(y))
            l_guide = calculate_structural_guidance_loss(hidden, scaffold)
            l_total = l_pred + lambda_guidance * l_guide
            l_total.backward()
            optimizer.step()
            total_loss += l_total.item()
            total_pred += l_pred.item()
            total_guide += l_guide.item()
        
        lr_scheduler.step()
        
        # --- 验证 ---
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in data['val_loader']:
                x, y = x.to(device), y.to(device)
                pred, _, _ = model(x)
                val_preds.append(pred)
                val_trues.append(y)
        
        val_preds = torch.cat(val_preds, dim=0)
        val_trues = torch.cat(val_trues, dim=0)
        val_preds_rescaled = scaler.inverse_transform(val_preds)
        val_trues_rescaled = scaler.inverse_transform(val_trues)
        val_rmse = masked_rmse(val_preds_rescaled, val_trues_rescaled).item()

        print(f"Epoch {epoch+1:03d} | Train Loss: {total_loss/len(data['train_loader']):.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/ScaffNet_best.pth")
            print("  Validation RMSE improved. Model saved.")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break
            
    # --- 测试 ---
    print("\nTraining finished. Evaluating on test set...")
    model.load_state_dict(torch.load(f"{save_dir}/ScaffNet_best.pth"))
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in data['test_loader']:
            x, y = x.to(device), y.to(device)
            pred, _, _ = model(x)
            test_preds.append(pred)
            test_trues.append(y)
            
    test_preds = torch.cat(test_preds, dim=0)
    test_trues = torch.cat(test_trues, dim=0)
    test_preds_rescaled = scaler.inverse_transform(test_preds)
    test_trues_rescaled = scaler.inverse_transform(test_trues)

    print("------ Test Results ------")
    for horizon_i in [2, 5, 11]: # 对应第3, 6, 12个预测步
        mae = masked_mae(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        mape = masked_mape(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        rmse = masked_rmse(test_preds_rescaled[:, :, horizon_i, :], test_trues_rescaled[:, :, horizon_i, :]).item()
        print(f"Horizon {horizon_i+1:02d} - MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}")

def main(args):
    # 1. 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 加载数据 (使用我们lib/utils.py中的加载器)
    print("加载数据...")
    data = load_dataset(
        dataset_dir=config['base_dir'], 
        batch_size=config['batch_size'], 
        test_batch_size=config['test_batch_size']
    )
    scaler = data['scaler']

    # 4. 初始化模型 (直接实例化ScaffNet)
    print("初始化模型...")
    model_config = {k: v for k, v in config.items() if k not in ['base_dir', 'batch_size', 'test_batch_size', 'epochs', 'patience', 'learning_rate', 'lambda_guidance', 'lr_decay_rate', 'lr_decay_step']}
    model = ScaffNet(**model_config).to(device)
    
    # 5. 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    prediction_loss_fn = masked_mae
    
    # 6. 直接调用我们定义的训练和评估流程
    print(f"开始训练 {model.__class__.__name__} on {config['base_dir'].split('/')[-1]}...")
    run(model, data, optimizer, lr_scheduler, prediction_loss_fn, config['lambda_guidance'], scaler, device, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/scaffnet_pems04.yaml', type=str, help='Config file path')
    args = parser.parse_args()
    main(args) 