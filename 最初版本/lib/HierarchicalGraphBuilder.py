import networkx as nx
import numpy as np
import torch
import pandas as pd

class HierarchicalGraphBuilder:
    """
    基于BCC算法构建层级图
    输入: 原始图的邻接矩阵 (numpy array or sparse matrix)
    输出: 区域图的邻接矩阵和原始-区域映射矩阵
    """
    def __init__(self, adj_mx_o):
        self.adj_mx_o = adj_mx_o
        self.num_nodes_o = adj_mx_o.shape[0]
        self.graph_o = nx.from_numpy_array(adj_mx_o)

    def build(self):
        # 1. 计算双连通分量 (BCC) 和割点
        bccs = list(nx.biconnected_components(self.graph_o))
        cut_vertices = set(nx.articulation_points(self.graph_o))
        self.num_regions_r = len(bccs)

        print(f"Original graph has {self.num_nodes_o} nodes.")
        print(f"Found {self.num_regions_r} regions (BCCs).")
        print(f"Found {len(cut_vertices)} cut vertices.")

        # 2. 构建原始-区域映射矩阵 M_or
        # M_or (N_o x N_r), a soft assignment matrix
        M_or = torch.zeros((self.num_nodes_o, self.num_regions_r))
        
        node_to_idx_o = {node: i for i, node in enumerate(self.graph_o.nodes())}

        for r_idx, bcc in enumerate(bccs):
            for node in bcc:
                o_idx = node_to_idx_o[node]
                M_or[o_idx, r_idx] = 1
        
        # 归一化，使得每个节点属于区域的权重和为1（对于割点）
        row_sums = M_or.sum(axis=1)
        row_sums[row_sums == 0] = 1 # avoid division by zero
        M_or = M_or / row_sums[:, np.newaxis]

        # 3. 构建区域图邻接矩阵 A_r
        # A_r = M_or^T * A_o * M_or
        A_r = torch.sparse.mm(torch.from_numpy(self.adj_mx_o).float().to_sparse(), M_or)
        A_r = torch.matmul(M_or.T, A_r)
        
        # 将对角线元素设为0，并进行二值化
        A_r.fill_diagonal_(0)
        A_r = (A_r > 0).float()

        return self.adj_mx_o, A_r, M_or

def get_adjacency_matrix(distance_df, num_of_vertices, id_filename=None):
    """
    从distance.csv构建邻接矩阵
    """
    if 'from' in distance_df.columns:
        distance_df = distance_df.rename(columns={'from': 'from_node', 'to': 'to_node'})
    
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    
    for row in distance_df.values:
        if len(row) != 3:
            continue
        i, j = int(row[0]), int(row[1])
        A[i, j] = 1
        A[j, i] = 1
    
    return A

if __name__ == '__main__':
    # 加载distance.csv
    distance_df = pd.read_csv('data/PEMS08/distance.csv')
    num_nodes = 170  # PEMS08的节点数
    
    # 构建原始邻接矩阵
    adj_mx_o = get_adjacency_matrix(distance_df, num_nodes)
    
    # 构建层级图
    builder = HierarchicalGraphBuilder(adj_mx_o)
    _, A_r, M_or = builder.build()
    
    # 保存结果
    torch.save(A_r, 'data/PEMS08/adj_mx_r.pt')
    torch.save(M_or, 'data/PEMS08/M_or.pt')
    print("Hierarchical graph files saved to data/PEMS08/") 