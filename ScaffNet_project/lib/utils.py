# ===============================================================================
# 文件路径: lib/utils.py
# 描述: 提供了项目所需的全部核心工具函数。
# ===============================================================================
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

class StandardScaler:
    """标准的Z-Score归一化器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=data.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=data.device)
        return (data * std) + mean

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    """从AGCRN-master的.npz数据格式加载数据"""
    data = {}
    print(f"从目录 {dataset_dir} 加载数据...")
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    data['train_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_train']).float(), torch.from_numpy(data['y_train']).float()),
        batch_size=batch_size, shuffle=True
    )
    data['val_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_val']).float(), torch.from_numpy(data['y_val']).float()),
        batch_size=test_batch_size if test_batch_size else batch_size, shuffle=False
    )
    data['test_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).float()),
        batch_size=test_batch_size if test_batch_size else batch_size, shuffle=False
    )
    data['scaler'] = scaler
    print(f"数据加载完成。训练集样本数: {len(data['x_train'])}")
    return data

def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.div(labels - preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) 