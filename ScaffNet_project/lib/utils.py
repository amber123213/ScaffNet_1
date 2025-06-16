# 文件路径: lib/utils.py
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

class StandardScaler:
    """标准Z-Score归一化"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.mean.device)
        return (data * self.std) + self.mean

def load_dataset_from_npz(dataset_dir, batch_size, test_batch_size=None):
    """
    一个完整的、自给自足的数据加载函数。
    它会读取AGCRN格式的.npz文件，进行归一化，并创建DataLoader。
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    # 只使用训练集数据计算mean和std
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    # 对所有数据集应用归一化
    for category in ['train', 'val', 'test']:
        # 只归一化第一个特征（例如，速度或流量）
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        # 如果y也有特征需要归一化，也应在此处理，但通常y的归一化使用相同的scaler
        # data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    # 将numpy数组转换为torch张量并创建DataLoader
    data['train_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_train']).float(), torch.from_numpy(data['y_train']).float()),
        batch_size=batch_size,
        shuffle=True
    )
    data['val_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_val']).float(), torch.from_numpy(data['y_val']).float()),
        batch_size=test_batch_size if test_batch_size else batch_size,
        shuffle=False
    )
    data['test_loader'] = DataLoader(
        TensorDataset(torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).float()),
        batch_size=test_batch_size if test_batch_size else batch_size,
        shuffle=False
    )
    data['scaler'] = scaler
    # 保留原始测试集y，用于最终评估
    data['y_test_raw'] = torch.from_numpy(data['y_test']).float()
    
    return data

def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))

def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = torch.abs(torch.div(labels - preds, labels))
    loss = loss * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)) * 100

def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))
