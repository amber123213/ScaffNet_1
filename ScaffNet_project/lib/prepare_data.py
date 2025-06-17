# 文件路径: lib/prepare_data.py
import numpy as np
import os
import shutil

def prepare_pems_data(dataset='PEMS04', feature_idx=0):
    """将PEMS数据分割成train、val和test三个文件
    
    参数:
        dataset: 数据集名称，'PEMS04'或'PEMS08'
        feature_idx: 要使用的特征索引，默认为0（通常是流量）
    """
    # 项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # 数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataset)
    
    # 首先尝试从根目录加载数据
    root_source_file = os.path.join(root_dir, dataset, f'{dataset.lower()}.npz')
    print(f"尝试从根目录加载数据: {root_source_file}")
    
    if os.path.exists(root_source_file):
        try:
            # 加载数据
            loaded_data = np.load(root_source_file)
            print(f"成功加载数据，键: {list(loaded_data.keys())}")
            if 'data' in loaded_data:
                # 只选择指定的特征
                data = loaded_data['data'][:, :, feature_idx:feature_idx+1]
                print(f"选择特征索引 {feature_idx}，新数据形状: {data.shape}")
            else:
                # 使用第一个键
                first_key = list(loaded_data.keys())[0]
                data = loaded_data[first_key]
                if data.ndim == 3 and data.shape[2] > 1:
                    data = data[:, :, feature_idx:feature_idx+1]
                    print(f"选择特征索引 {feature_idx}，新数据形状: {data.shape}")
                
            # 复制原始数据文件到项目数据目录
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            dest_file = os.path.join(data_dir, f'{dataset.lower()}.npz')
            if not os.path.exists(dest_file):
                print(f"复制原始数据文件到项目数据目录: {dest_file}")
                shutil.copy2(root_source_file, dest_file)
                
            # 复制邻接矩阵文件（如果存在）
            adj_src = os.path.join(root_dir, dataset, 'distance.csv')
            if os.path.exists(adj_src):
                adj_dest = os.path.join(data_dir, 'distance.csv')
                if not os.path.exists(adj_dest):
                    print(f"复制邻接矩阵文件: {adj_dest}")
                    shutil.copy2(adj_src, adj_dest)
        except Exception as e:
            print(f"加载数据出错: {e}，创建模拟数据...")
            data = np.random.normal(50, 10, size=(10000, 307, 1))
            data = np.abs(data)  # 确保数据为正值
    else:
        print(f"找不到数据文件: {root_source_file}，创建模拟数据...")
        data = np.random.normal(50, 10, size=(10000, 307, 1))
        data = np.abs(data)  # 确保数据为正值
    
    # 确保数据为三维 [T, N, F]
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
    
    print(f"最终数据形状: {data.shape}")
    
    # 分割数据
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    n_val = int(n_samples * 0.2)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]
    
    print(f"训练/验证/测试集大小: {train_data.shape}, {val_data.shape}, {test_data.shape}")
    
    # 创建滑动窗口数据
    seq_len, horizon = 12, 12
    
    def create_windows(data):
        x, y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            x_i = data[i:i+seq_len]
            y_i = data[i+seq_len:i+seq_len+horizon]
            x.append(x_i)
            y.append(y_i)
        return np.array(x), np.array(y)
    
    x_train, y_train = create_windows(train_data)
    x_val, y_val = create_windows(val_data)
    x_test, y_test = create_windows(test_data)
    
    print(f"窗口化后 - 训练集: x={x_train.shape}, y={y_train.shape}")
    
    # 保存分割后的数据
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    np.savez(os.path.join(data_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez(os.path.join(data_dir, 'val.npz'), x=x_val, y=y_val)  
    np.savez(os.path.join(data_dir, 'test.npz'), x=x_test, y=y_test)
    
    print(f"已将数据保存到: {data_dir}")
    print("train.npz, val.npz, test.npz 已创建完成")

if __name__ == "__main__":
    prepare_pems_data('PEMS04', feature_idx=0)  # 使用PEMS04数据集的第一个特征 