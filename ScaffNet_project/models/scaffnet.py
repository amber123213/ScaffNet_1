# 文件路径: models/scaffnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# AGCRNCell中的图卷积部分被替换，但我们保留了它的基本GRU结构思想
# 因此，为了简化，我们将所有逻辑都包含在ScaffNet内部，不再需要单独的AGCN.py和AGCRNCell.py

class ScaffoldingGraphLearner(nn.Module):
    """训练脚手架学习器 (TSL)"""
    def __init__(self, num_nodes, embedding_dim):
        super(ScaffoldingGraphLearner, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)

    def forward(self):
        adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
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

    def forward(self, x_t, h_prev, A_infer):
        combined_input = torch.cat([x_t, h_prev], dim=-1)
        z_t = torch.sigmoid(self.update_gcn(combined_input, A_infer))
        r_t = torch.sigmoid(self.reset_gcn(combined_input, A_infer))
        combined_candidate = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_tilde = torch.tanh(self.candidate_gcn(combined_candidate, A_infer))
        return (1.0 - z_t) * h_prev + z_t * h_tilde

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
        
        # 从AGCRN的实现中借鉴，使用一个简单的线性层作为输出
        self.output_mlp = nn.Linear(self.hidden_dim, output_dim * horizon)

    def forward(self, x_history):
        # x_history shape: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, _, _ = x_history.shape
        device = x_history.device

        A_infer = torch.eye(self.num_nodes, device=device)
        h_t = torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
        
        hidden_states_history = [] if self.training else None
        
        for t in range(seq_len):
            h_t = self.gru_cell(x_history[:, t, :, :], h_t, A_infer)
            if self.training:
                hidden_states_history.append(h_t)
        
        prediction = self.output_mlp(h_t)
        prediction = prediction.view(batch_size, self.num_nodes, self.horizon, self.output_dim)
        
        if self.training:
            A_scaffold = self.scaffolding_learner()
            hidden_states_history = torch.stack(hidden_states_history, dim=1)
            return prediction, hidden_states_history, A_scaffold
        else:
            # 保持返回值的数量一致，避免在训练脚本中写if/else
            return prediction, None, None
