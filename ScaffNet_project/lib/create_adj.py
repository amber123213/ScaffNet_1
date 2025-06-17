import numpy as np
import pickle
import os
import pandas as pd

def create_adjacency_matrix(dataset='PEMS04'):
    """
    从distance.csv文件创建邻接矩阵
    
    参数:
        dataset: 数据集名称，'PEMS04'或'PEMS08'
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataset)
    distance_file = os.path.join(data_dir, 'distance.csv')
    
    if not os.path.exists(distance_file):
        print(f"距离文件不存在: {distance_file}")
        return False
    
    print(f"从文件创建邻接矩阵: {distance_file}")
    
    # 读取CSV文件（边列表格式）
    df = pd.read_csv(distance_file)
    print(f"读取了 {len(df)} 条边")
    
    # 获取节点数量
    if dataset == 'PEMS04':
        num_nodes = 307
    elif dataset == 'PEMS08':
        num_nodes = 170
    else:
        # 从数据中推断节点数量
        all_nodes = set(df['from'].unique()) | set(df['to'].unique())
        num_nodes = max(all_nodes) + 1
    
    print(f"节点数量: {num_nodes}")
    
    # 创建空的邻接矩阵
    adj_mx = np.zeros((num_nodes, num_nodes))
    
    # 填充邻接矩阵
    for _, row in df.iterrows():
        from_node = int(row['from'])
        to_node = int(row['to'])
        distance = float(row['cost'])
        
        # 使用高斯核将距离转换为相似度
        sigma2 = 10.0
        similarity = np.exp(-distance**2/sigma2)
        
        # 填充邻接矩阵（双向）
        adj_mx[from_node, to_node] = similarity
        adj_mx[to_node, from_node] = similarity
    
    # 确保对角线为1（自环）
    np.fill_diagonal(adj_mx, 1.0)
    
    print(f"邻接矩阵形状: {adj_mx.shape}")
    print(f"邻接矩阵非零元素数量: {np.count_nonzero(adj_mx)}")
    
    # 创建DCRNN格式的邻接矩阵
    adj_mx_tuple = [('scaled_identity', 0), ('scaled_laplacian', 0), adj_mx]
    
    # 保存为pickle格式
    with open(os.path.join(data_dir, 'adj_mx.pkl'), 'wb') as f:
        pickle.dump(adj_mx_tuple, f)
    print(f"邻接矩阵已保存为pickle格式: {os.path.join(data_dir, 'adj_mx.pkl')}")
    
    # 保存为npz格式
    np.savez(os.path.join(data_dir, 'adj_mx.npz'), adj_mx=adj_mx)
    print(f"邻接矩阵已保存为npz格式: {os.path.join(data_dir, 'adj_mx.npz')}")
    
    return True

if __name__ == "__main__":
    create_adjacency_matrix('PEMS04') 