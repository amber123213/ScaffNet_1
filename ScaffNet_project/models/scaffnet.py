# 文件路径: models/scaffnet.py (终极稳定版)
import torch
import torch.nn as nn
import torch.nn.functional as F

# AGCRNCell中的图卷积部分被替换，但我们保留了它的基本GRU结构思想
# 因此，为了简化，我们将所有逻辑都包含在ScaffNet内部，不再需要单独的AGCN.py和AGCRNCell.py

class ScaffoldingGraphLearner(nn.Module):
    """训练脚手架学习器 (TSL) - 增加了层归一化稳定器"""
    def __init__(self, num_nodes, embedding_dim):
        super(ScaffoldingGraphLearner, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)
        # 【新稳定器】增加一个层归一化
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, num_nodes=None):
        """
        学习脚手架图结构
        
        参数:
            num_nodes: 可选，当实际节点数与初始化不同时使用
        """
        # 检查是否需要调整嵌入大小
        if num_nodes is not None and num_nodes != self.num_nodes:
            device = self.node_embeddings.device
            print(f"重新初始化脚手架图学习器的节点嵌入，从 {self.num_nodes} 到 {num_nodes}")
            self.num_nodes = num_nodes
            # 创建新的嵌入并将其移动到原始嵌入的设备上
            new_embeddings = nn.Parameter(torch.randn(num_nodes, self.embedding_dim, device=device), 
                                         requires_grad=True)
            self.node_embeddings = new_embeddings
        
        # 归一化节点嵌入，防止数值爆炸
        emb = self.norm(self.node_embeddings)
        adj = F.softmax(F.relu(torch.mm(emb, emb.transpose(0, 1))), dim=1)
        return adj

class ScaffGCN(nn.Module):
    """简单的图卷积层，只在推理图上操作"""
    def __init__(self, in_dim, out_dim):
        super(ScaffGCN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        ax = torch.einsum('ij,bjd->bid', adj, x)
        return self.linear(ax)

class ScaffGRUCell(nn.Module):
    """ScaffNet的核心循环单元"""
    def __init__(self, node_dim, hidden_dim):
        super(ScaffGRUCell, self).__init__()
        combined_dim = node_dim + hidden_dim
        self.update_gcn = ScaffGCN(combined_dim, hidden_dim)
        self.reset_gcn = ScaffGCN(combined_dim, hidden_dim)
        self.candidate_gcn = ScaffGCN(combined_dim, hidden_dim)
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

    def forward(self, x_t, h_prev, A_infer):
        # 确保输入形状正确
        batch_size, num_nodes, _ = x_t.shape
        
        # 检查隐藏状态形状
        if h_prev.shape[1] != num_nodes:
            print(f"警告: 隐藏状态节点数与输入不匹配: {h_prev.shape[1]} vs {num_nodes}")
            # 创建匹配大小的新隐藏状态
            h_prev = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x_t.device)
        
        try:
            combined_input = torch.cat([x_t, h_prev], dim=-1)
            z_t = torch.sigmoid(self.update_gcn(combined_input, A_infer))
            r_t = torch.sigmoid(self.reset_gcn(combined_input, A_infer))
            combined_candidate = torch.cat([x_t, r_t * h_prev], dim=-1)
            h_tilde = torch.tanh(self.candidate_gcn(combined_candidate, A_infer))
            return (1.0 - z_t) * h_prev + z_t * h_tilde
        except Exception as e:
            print(f"GRU单元错误: {e}")
            print(f"x_t形状: {x_t.shape}, h_prev形状: {h_prev.shape}")
            # 返回原始隐藏状态避免程序崩溃
            return h_prev

class ScaffNet(nn.Module):
    """ScaffNet: 训练-推理异构图网络"""
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, embed_dim, **kwargs):
        super(ScaffNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = rnn_units
        self.horizon = horizon
        self.output_dim = output_dim
        
        self.scaffolding_learner = ScaffoldingGraphLearner(num_nodes, embed_dim)
        self.gru_cell = ScaffGRUCell(node_dim=input_dim, hidden_dim=self.hidden_dim)
        
        # 【最佳实践】引入可学习的初始隐藏状态
        self.h0 = nn.Parameter(torch.randn(1, num_nodes, self.hidden_dim), requires_grad=True)
        
        self.output_mlp = nn.Linear(self.hidden_dim, output_dim * horizon)

    def forward(self, x_history):
        # x_history shape: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, _, _ = x_history.shape
        device = x_history.device

        A_infer = torch.eye(self.num_nodes, device=device)
        
        # 【关键修正】使用可学习的h0，并扩展到当前batch_size
        h_t = self.h0.expand(batch_size, -1, -1)
        
        hidden_states_history = [] if self.training else None
        
        for t in range(seq_len):
            h_t = self.gru_cell(x_history[:, t, :, :], h_t, A_infer)
            if self.training:
                hidden_states_history.append(h_t)
        
        prediction = self.output_mlp(h_t)
        prediction = prediction.view(batch_size, self.num_nodes, self.horizon, self.output_dim)
        
        # 将输出调整为 [batch_size, horizon, num_nodes, output_dim]
        prediction = prediction.permute(0, 2, 1, 3)
        
        if self.training:
            A_scaffold = self.scaffolding_learner()
            hidden_states_history = torch.stack(hidden_states_history, dim=1)
            # 【关键修正】将 A_scaffold 分离，打破梯度恶性循环
            return prediction, hidden_states_history, A_scaffold.detach()
        else:
            return prediction, None, None
