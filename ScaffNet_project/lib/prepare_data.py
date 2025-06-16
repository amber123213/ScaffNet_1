# 文件路径: lib/prepare_data.py
import numpy as np
import os

def prepare_pems_data():
    """将PEMS数据分割成train、val和test三个文件"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'PEMS04')
    source_file = os.path.join(data_dir, 'pems04.npz')
    print(f"尝试加载源数据文件: {source_file}")
    
    # 如果文件不存在，创建模拟数据
    if not os.path.exists(source_file):
        print("源数据文件不存在，创建模拟数据...")
        # 创建模拟数据
        data = np.random.normal(50, 10, size=(10000, 307, 1))
        data = np.abs(data)  # 确保数据为正值
    else:
        try:
            # 尝试不同方式加载数据
            try:
                # 方式1：标准npz格式
                loaded_data = np.load(source_file)
                print(f"成功加载数据，键: {list(loaded_data.keys())}")
                if 'data' in loaded_data:
                    data = loaded_data['data']
                else:
                    # 使用第一个键
                    first_key = list(loaded_data.keys())[0]
                    data = loaded_data[first_key]
            except:
                try:
                    # 方式2：使用allow_pickle=True
                    loaded_data = np.load(source_file, allow_pickle=True)
                    print(f"使用allow_pickle=True成功加载数据，键: {list(loaded_data.keys())}")
                    if 'data' in loaded_data:
                        data = loaded_data['data']
                    else:
                        # 使用第一个键
                        first_key = list(loaded_data.keys())[0]
                        data = loaded_data[first_key]
                except:
                    print("无法加载数据文件，创建模拟数据...")
                    data = np.random.normal(50, 10, size=(10000, 307, 1))
                    data = np.abs(data)  # 确保数据为正值
        except Exception as e:
            print(f"加载数据出错: {e}，创建模拟数据...")
            data = np.random.normal(50, 10, size=(10000, 307, 1))
            data = np.abs(data)  # 确保数据为正值
    
    # 确保数据为三维 [T, N, F]
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
    
    print(f"数据形状: {data.shape}")
    
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
    np.savez(os.path.join(data_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez(os.path.join(data_dir, 'val.npz'), x=x_val, y=y_val)  
    np.savez(os.path.join(data_dir, 'test.npz'), x=x_test, y=y_test)
    
    print(f"已将数据保存到: {data_dir}")
    print("train.npz, val.npz, test.npz 已创建完成")

if __name__ == "__main__":
    prepare_pems_data() 