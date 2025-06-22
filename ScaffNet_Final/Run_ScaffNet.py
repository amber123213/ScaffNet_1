import argparse
import torch
import numpy as np
import os
import sys
import configparser

# 将项目根目录添加到路径
sys.path.append(os.path.join(os.getcwd()))
from lib.dataloader import get_dataloader
from lib.metrics import MAE_torch, MAPE_torch, RMSE_torch
from lib.logger import get_logger
from model.AGCRN import AGCRN
from model.scaffnet import ScaffNet # 确保导入我们的ScaffNet
from model.BasicTrainer import Trainer as BasicTrainer

def get_model(args):
    if args.model_name == 'AGCRN':
        model = AGCRN(args).to(args.device)
    elif args.model_name == 'ScaffNet':
        model = ScaffNet(args).to(args.device)
    else:
        raise ValueError
    return model

if __name__ == '__main__':
    # ----------------- 1. 参数解析 -----------------
    # 首先创建解析器
    parser = argparse.ArgumentParser()
    
    # 从配置文件中读取默认值
    # 这是AGCRN-master原有的逻辑，我们保留它
    config_filename = 'model/PEMSD4_AGCRN.conf' # 默认配置文件
    config = configparser.ConfigParser()
    try:
        config.read(config_filename)
        print(f"读取默认配置文件: {config_filename}")
    except:
        # 创建一个空的配置
        config['model'] = {}
        config['data'] = {}
        config['train'] = {}
    
    # 定义所有参数
    parser.add_argument('--model_name', type=str, default=config['model'].get('model_name', 'AGCRN'), help='模型名称')
    parser.add_argument('--batch_size', type=int, default=config['data'].getint('batch_size', 64))
    parser.add_argument('--input_dim', type=int, default=config['model'].getint('input_dim', 2))
    parser.add_argument('--output_dim', type=int, default=config['model'].getint('output_dim', 1))
    parser.add_argument('--rnn_units', type=int, default=config['model'].getint('rnn_units', 64))
    parser.add_argument('--num_nodes', type=int, default=config['model'].getint('num_nodes', 207))
    parser.add_argument('--cheb_k', type=int, default=config['model'].getint('cheb_k', 2))
    parser.add_argument('--embed_dim', type=int, default=config['model'].getint('embed_dim', 10))
    parser.add_argument('--horizon', type=int, default=config['model'].getint('horizon', 12))
    parser.add_argument('--seq_len', type=int, default=config['model'].getint('seq_len', 12))
    parser.add_argument('--loss_func', type=str, default=config['train'].get('loss_func', 'mae'))
    parser.add_argument('--seed', type=int, default=config['train'].getint('seed', 10))
    parser.add_argument('--epochs', type=int, default=config['train'].getint('epochs', 200))
    parser.add_argument('--learning_rate', type=float, default=config['train'].getfloat('learning_rate', 0.001))
    parser.add_argument('--patience', type=int, default=config['train'].getint('patience', 20))
    parser.add_argument('--optimizer', type=str, default=config['train'].get('optimizer', 'adam'))
    parser.add_argument('--dataset_dir', type=str, default=config['data'].get('dataset_dir', 'data/PEMS04'))
    parser.add_argument('--dataset', type=str, default='PEMS04')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--real_value', type=bool, default=True)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--grad_norm', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=int, default=5)
    parser.add_argument('--teacher_forcing', type=bool, default=False)
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--mae_thresh', type=float, default=None)
    parser.add_argument('--mape_thresh', type=float, default=0.0)
    parser.add_argument('--debug', type=bool, default=False)
    
    # ScaffNet的专属参数
    parser.add_argument('--lambda_guidance', type=float, default=config['train'].getfloat('lambda_guidance', 0.01))
    
    # 命令行参数会覆盖配置文件中的值
    parser.add_argument('--config_filename', type=str, default='model/PEMSD4_ScaffNet.conf', help='Configuration file name.')
    
    # 在定义完所有参数后，才进行解析
    args = parser.parse_args()
    
    # 如果命令行指定了新的配置文件，则重新加载它
    if args.config_filename:
        print(f"读取指定配置文件: {args.config_filename}")
        new_config = configparser.ConfigParser()
        new_config.read(args.config_filename)
        
        # 将配置文件中的值更新到args中
        if 'data' in new_config:
            for key, value in new_config['data'].items():
                if hasattr(args, key):
                    # 尝试转换类型
                    try:
                        orig_value = getattr(args, key)
                        if isinstance(orig_value, bool):
                            value = value.lower() in ['true', 'yes', '1']
                        elif isinstance(orig_value, int):
                            value = int(value)
                        elif isinstance(orig_value, float):
                            value = float(value)
                        setattr(args, key, value)
                    except:
                        print(f"无法转换参数 {key} 的值: {value}")
        
        if 'model' in new_config:
            for key, value in new_config['model'].items():
                if hasattr(args, key):
                    try:
                        orig_value = getattr(args, key)
                        if isinstance(orig_value, bool):
                            value = value.lower() in ['true', 'yes', '1']
                        elif isinstance(orig_value, int):
                            value = int(value)
                        elif isinstance(orig_value, float):
                            value = float(value)
                        setattr(args, key, value)
                    except:
                        print(f"无法转换参数 {key} 的值: {value}")
                        
        if 'train' in new_config:
            for key, value in new_config['train'].items():
                if hasattr(args, key):
                    try:
                        orig_value = getattr(args, key)
                        if isinstance(orig_value, bool):
                            value = value.lower() in ['true', 'yes', '1']
                        elif isinstance(orig_value, int):
                            value = int(value)
                        elif isinstance(orig_value, float):
                            value = float(value)
                        setattr(args, key, value)
                    except:
                        print(f"无法转换参数 {key} 的值: {value}")

    # ----------------- 2. 环境与数据准备 -----------------
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger('./experiments', __name__, 'info.log')
    logger.info(str(args))
    
    # 设置与数据集相关的参数
    args.val_ratio = 0.2
    args.test_ratio = 0.2
    args.lag = 12
    args.horizon = 12
    args.column_wise = False
    
    # 设置日志目录
    args.log_dir = './experiments/ScaffNet_{}'.format(args.dataset)
    args.model = args.model_name  # 用于兼容BasicTrainer
    args.debug = False
    args.tf_decay_steps = 2000  # teacher forcing decay steps
    args.lr_decay = False  # 学习率衰减
    
    train_loader, val_loader, test_loader, scaler = get_dataloader(args)
    
    # ----------------- 3. 模型构建与训练 -----------------
    model = get_model(args)
    
    # 使用L1Loss作为基础损失函数
    if args.loss_func == 'mae':
        loss_func = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_func = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_func}")
    
    # 创建优化器
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # 创建训练器
    if args.model_name == 'ScaffNet':
        # 对于ScaffNet，使用字符串标记特殊的损失计算方式
        trainer = BasicTrainer(model, 'scaffnet_loss', optimizer, 
                               train_loader, val_loader, test_loader, 
                               scaler, args, lr_scheduler=None)
    else:
        trainer = BasicTrainer(model, loss_func, optimizer, 
                               train_loader, val_loader, test_loader, 
                               scaler, args, lr_scheduler=None)
    
    # 开始训练
    trainer.train() 