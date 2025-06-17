import os
import argparse
import numpy as np
import pickle

def generate_train_val_test(args):
    # 加载原始数据
    data = np.load(args.traffic_df_filename)
    data = data['data']  # 假设数据在'data'键下
    print(f"原始数据形状: {data.shape}")
    
    # 数据预处理
    # 假设数据形状为 (time_steps, num_nodes, features)
    time_len = data.shape[0]
    num_samples = time_len - (args.seq_length + args.horizon) + 1
    
    train_ratio = 0.6
    test_ratio = 0.2
    
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    val_size = num_samples - train_size - test_size
    
    print(f"训练样本数: {train_size}, 验证样本数: {val_size}, 测试样本数: {test_size}")
    
    x_offsets = np.sort(np.arange(-(args.seq_length - 1), 1, 1))
    y_offsets = np.sort(np.arange(1, args.horizon + 1, 1))
    
    # 生成样本
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(time_len - abs(max(y_offsets)))
    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    print(f"x形状: {x.shape}, y形状: {y.shape}")
    
    # 划分训练、验证和测试集
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[-test_size:], y[-test_size:]
    
    # 保存数据
    for cat in ["train", "val", "test"]:
        _x, _y = locals()[f"x_{cat}"], locals()[f"y_{cat}"]
        print(f"{cat} x: {_x.shape}, y: {_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
        )
        print(f"{cat}.npz 已保存")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--traffic_df_filename", type=str, required=True, help="交通数据文件路径")
    parser.add_argument("--seq_length", type=int, default=12, help="输入序列长度")
    parser.add_argument("--horizon", type=int, default=12, help="预测序列长度")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    generate_train_val_test(args)

if __name__ == "__main__":
    main() 