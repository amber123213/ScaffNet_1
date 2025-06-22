import numpy as np
import argparse
import os

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/PEMS04/pems04.npz', help='data file')
    parser.add_argument('--output_dir', type=str, default='data/PEMS04', help='output directory')
    parser.add_argument('--window', type=int, default=12, help='window size')
    parser.add_argument('--horizon', type=int, default=12, help='horizon')
    args = parser.parse_args()
    
    # 加载数据
    data = np.load(args.data_file)
    for k in data.keys():
        print(f"Key: {k}, Shape: {data[k].shape}")
    
    # 假设数据中有一个名为'data'的键
    if 'data' in data.keys():
        traffic_data = data['data']
        print(f"Processing data with shape: {traffic_data.shape}")
        
        # 应用滑动窗口
        X, Y = Add_Window_Horizon(traffic_data, window=args.window, horizon=args.horizon)
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # 保存处理后的数据
        output_file = os.path.join(args.output_dir, f"processed_data_w{args.window}_h{args.horizon}.npz")
        np.savez(output_file, x=X, y=Y)
        print(f"Data saved to {output_file}")
    else:
        print(f"Error: 'data' key not found in {args.data_file}. Available keys: {data.keys()}")


