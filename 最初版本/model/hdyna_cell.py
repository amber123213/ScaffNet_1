import torch
import torch.nn as nn
import torch.nn.functional as F

# 动态因果记忆网络 (DCMN)
class DynamicCausalMemoryNetwork(nn.Module):
    def __init__(self, num_nodes, node_embed_dim, pattern_dim, M_global, S_global, M_local, S_local, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_dim = node_embed_dim
        self.pattern_dim = pattern_dim
        self.M_global = M_global
        self.S_global = S_global
        self.M_local = M_local
        self.S_local = S_local
        self.hidden_dim = hidden_dim

        # 可学习的记忆库
        self.global_memory_bank = nn.Parameter(torch.randn(M_global, S_global, pattern_dim))
        self.local_memory_bank = nn.Parameter(torch.randn(M_local, S_local, pattern_dim))
        
        # 特征到模式空间的映射
        self.feature_to_pattern_mlp = nn.Linear(hidden_dim, pattern_dim)
        
        # 节点自适应参数学习
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        # weight_pool将融合后的上下文映射回隐藏维度
        self.weight_pool = nn.Parameter(torch.randn(node_embed_dim, pattern_dim * 2, hidden_dim))

    def forward(self, x_hist):
        # x_hist: (batch_size, num_nodes, history_len, hidden_dim)
        query_pattern = self.feature_to_pattern_mlp(x_hist) # (B, N, S, P_dim)

        # 模式匹配 (用点积注意力简化DTW)
        query_flat = query_pattern.reshape(-1, self.S_local, self.pattern_dim)
        
        # Local
        sim_local = torch.einsum('bsp,msp->bnm', query_flat, self.local_memory_bank)
        attn_local = F.softmax(sim_local, dim=-1) # (B*N, M_local)
        
        # Global
        query_for_global = F.adaptive_avg_pool1d(query_flat.transpose(1, 2), self.S_global).transpose(1, 2)
        sim_global = torch.einsum('bsp,msp->bnm', query_for_global, self.global_memory_bank)
        attn_global = F.softmax(sim_global, dim=-1) # (B*N, M_global)
        
        # 上下文读取
        local_context = torch.einsum('bnm,mpd->bnd', attn_local, self.local_memory_bank.mean(dim=1))
        global_context = torch.einsum('bnm,mpd->bnd', attn_global, self.global_memory_bank.mean(dim=1))

        # 融合与自适应
        fused_context = torch.cat([local_context, global_context], dim=-1)
        fused_context = fused_context.view(-1, self.num_nodes, self.pattern_dim * 2)
        
        node_specific_weights = torch.einsum('nd,dfh->nfh', self.node_embedding, self.weight_pool)
        final_context = torch.einsum('bnd,ndh->bnh', fused_context, node_specific_weights)
        
        return final_context

# H-DYNA 循环单元
class H_DYNA_Cell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, node_embed_dim, M_global, S_global, M_local, S_local, pattern_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 将DCMN作为核心的空间建模器
        self.dcn = DynamicCausalMemoryNetwork(num_nodes, node_embed_dim, pattern_dim, 
                                            M_global, S_global, M_local, S_local, hidden_dim)
        
        # GRU的门控
        self.update_gate = nn.Linear(input_dim + hidden_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim + hidden_dim, hidden_dim)
        self.candidate_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        # x_t: (B, N, input_dim), h_prev: (B, N, hidden_dim)
        h_prev_as_query = h_prev.unsqueeze(2).expand(-1, -1, self.dcn.S_local, -1)
        
        context = self.dcn(h_prev_as_query) # (B, N, H_dim)
        
        combined_for_gates = torch.cat([x_t, context, h_prev], dim=-1)
        z_t = torch.sigmoid(self.update_gate(combined_for_gates))
        r_t = torch.sigmoid(self.reset_gate(combined_for_gates))
        
        combined_for_candidate = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_candidate = torch.tanh(self.candidate_gate(combined_for_candidate))
        
        h_next = (1.0 - z_t) * h_prev + z_t * h_candidate
        
        return h_next 