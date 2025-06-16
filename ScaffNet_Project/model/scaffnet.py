import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaffoldingGraphLearner(nn.Module):
    """训练脚手架学习器 (TSL)"""
    def __init__(self, num_nodes, embedding_dim):
        super(ScaffoldingGraphLearner, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)

    def forward(self):
        adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        return adj

class ScaffGCN(nn.Module):
    """简单的图卷积层"""
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
        self.hidden_dim = hidden_dim
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
        h_next = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_next

class ScaffNet(nn.Module):
    """ScaffNet: 训练-推理异构图网络"""
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, horizon, scaffold_embedding_dim=10, **kwargs):
        super(ScaffNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.output_dim = output_dim
        
        self.scaffolding_learner = ScaffoldingGraphLearner(num_nodes, scaffold_embedding_dim)
        self.gru_cell = ScaffGRUCell(node_dim=input_dim, hidden_dim=hidden_dim)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim * horizon)
        )

    def forward(self, batch):
        x_history = batch['x'] # Shape: (batch_size, seq_len, num_nodes, input_dim)
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
        prediction = prediction.view(batch_size, self.horizon, self.num_nodes, self.output_dim).transpose(1, 2)
        
        if self.training:
            A_scaffold = self.scaffolding_learner()
            hidden_states_history = torch.stack(hidden_states_history, dim=1)
            return prediction, hidden_states_history, A_scaffold
        else:
            return prediction, None, None 