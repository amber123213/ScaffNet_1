# ===============================================================================
# 文件路径: models/scaffnet.py (最终版)
# ===============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaffoldingGraphLearner(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(ScaffoldingGraphLearner, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)
    def forward(self):
        return F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

class ScaffGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ScaffGCN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        ax = torch.einsum('ij,bjd->bid', adj, x)
        return self.linear(ax)

class ScaffGRUCell(nn.Module):
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
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, embed_dim, **kwargs):
        super(ScaffNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = rnn_units
        self.horizon = horizon
        self.output_dim = output_dim
        self.scaffolding_learner = ScaffoldingGraphLearner(num_nodes, embed_dim)
        self.gru_cell = ScaffGRUCell(node_dim=input_dim, hidden_dim=self.hidden_dim)
        self.output_mlp = nn.Linear(self.hidden_dim, output_dim * horizon)

    def forward(self, x_history):
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
            return prediction, torch.stack(hidden_states_history, dim=1), self.scaffolding_learner()
        else:
            return prediction, None, None 