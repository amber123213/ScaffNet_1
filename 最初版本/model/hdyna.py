import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCausalMemoryNetwork(nn.Module):
    """
    动态因果记忆网络 (DCMN)
    改进自: PM-DMNet (记忆网络思想) + PDFormer (延迟感知思想)
    """
    def __init__(self, num_nodes, node_embed_dim, pattern_dim, M_global, S_global, M_local, S_local, hidden_dim, cheb_k):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.node_embed_dim = node_embed_dim
        self.pattern_dim = pattern_dim
        self.M_global = M_global
        self.M_local = M_local
        self.S_global = S_global # 模式长度
        self.S_local = S_local   # 历史查询片段长度

        # 可学习的记忆库 (Pattern-Matching)
        self.global_memory_bank = nn.Parameter(torch.randn(M_global, S_global, pattern_dim))
        self.local_memory_bank = nn.Parameter(torch.randn(M_local, S_local, pattern_dim))
        
        # 将输入历史映射到查询模式空间 (Delay-Aware)
        self.history_to_query_mlp = nn.Linear(hidden_dim, pattern_dim)

        # 节点自适应参数学习 (Node-Adaptive)
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, node_embed_dim), requires_grad=True)
        # 权重池将融合后的上下文[local, global]映射回隐藏维度
        self.weight_pool = nn.Parameter(torch.randn(node_embed_dim, pattern_dim * 2, hidden_dim)) 

    def forward(self, h_history):
        # h_history: (B, N, S_local, H_dim) - 代表过去S_local个时间步的隐藏状态
        batch_size = h_history.shape[0]

        # 1. 查询模式生成
        # (B, N, S, D) -> (B*N, S, D) -> (B*N, S, P_dim)
        query_pattern = self.history_to_query_mlp(h_history.reshape(-1, self.S_local, self.hidden_dim))

        # 2. 模式匹配 (使用点积注意力)
        # Local Memory
        sim_local = torch.einsum('bsp,msp->bm', query_pattern, self.local_memory_bank)
        attn_local = F.softmax(sim_local, dim=-1) # (B*N, M_local)
        
        # Global Memory
        # 为了匹配，将查询模式的长度从S_local调整到S_global
        if self.S_local != self.S_global:
            query_for_global = F.adaptive_avg_pool1d(query_pattern.transpose(1, 2), self.S_global).transpose(1, 2)
        else:
            query_for_global = query_pattern
        sim_global = torch.einsum('bsp,msp->bm', query_for_global, self.global_memory_bank)
        attn_global = F.softmax(sim_global, dim=-1) # (B*N, M_global)

        # 3. 上下文读取 (从记忆库中提取加权模式)
        local_context = torch.einsum('bm,msp->bsp', attn_local, self.local_memory_bank) # (B*N, S, P_dim)
        global_context = torch.einsum('bm,msp->bsp', attn_global, self.global_memory_bank) # (B*N, S, P_dim)
        
        # 取序列的平均作为最终上下文表示
        local_context = local_context.mean(dim=1) # (B*N, P_dim)
        global_context = global_context.mean(dim=1) # (B*N, P_dim)

        # 4. 融合与节点自适应变换
        fused_context = torch.cat([local_context, global_context], dim=-1) # (B*N, P_dim*2)
        fused_context = fused_context.view(batch_size, self.num_nodes, self.pattern_dim * 2) # (B, N, P_dim*2)
        
        node_specific_weights = torch.einsum('nd,dfh->nfh', self.node_embedding, self.weight_pool)
        final_context = torch.einsum('bnd,ndh->bnh', fused_context, node_specific_weights) # (B, N, H_dim)
        
        return final_context

class H_DYNA_Cell(nn.Module):
    """
    H-DYNA的核心循环单元
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, node_embed_dim, M_global, S_global, M_local, S_local, pattern_dim, cheb_k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dcn = DynamicCausalMemoryNetwork(num_nodes, node_embed_dim, pattern_dim, M_global, S_global, M_local, S_local, hidden_dim, cheb_k)
        
        # GRU门控机制
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate_gate = nn.Linear(input_dim + hidden_dim, hidden_dim) 

    def forward(self, x_t, h_prev, history_buffer):
        # x_t: (B, N, input_dim)
        # h_prev: (B, N, hidden_dim)
        # history_buffer: (B, N, S_local, hidden_dim)
        
        context = self.dcn(history_buffer)
        
        combined_for_update = torch.cat([x_t, h_prev], dim=-1)
        z_t = torch.sigmoid(self.update_gate(combined_for_update))
        
        combined_for_reset = torch.cat([x_t, h_prev], dim=-1)
        r_t = torch.sigmoid(self.reset_gate(combined_for_reset))
        
        combined_for_candidate = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_candidate = torch.tanh(self.candidate_gate(combined_for_candidate) + context)
        
        h_next = (1.0 - z_t) * h_prev + z_t * h_candidate
        
        return h_next

class H_DYNA(nn.Module):
    """
    主模型: H-DYNA
    """
    def __init__(self, args, adj_mx, adj_mx_r, M_or):
        super().__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.S_local = args.S_local
        
        # 保存图数据
        self.register_buffer('adj_mx', adj_mx)
        self.register_buffer('adj_mx_r', adj_mx_r)
        self.register_buffer('M_or', M_or)
        
        # 初始化H_DYNA_Cell
        self.hdyna_cell = H_DYNA_Cell(
            args.num_nodes, 
            self.input_dim, 
            self.hidden_dim, 
            args.embed_dim,
            args.M_global, 
            args.S_global, 
            args.M_local, 
            args.S_local, 
            args.pattern_dim,
            args.cheb_k
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, source, targets=None, teacher_forcing_ratio=0.5):
        batch_size, history_len, _, _ = source.shape
        
        # 初始化隐藏状态
        h_t = torch.zeros(batch_size, self.num_node, self.hidden_dim).to(source.device)
        history_buffer = torch.zeros(batch_size, self.num_node, self.S_local, self.hidden_dim).to(source.device)
        
        # Encoder
        for t in range(history_len):
            h_t = self.hdyna_cell(source[:, t, :, :], h_t, history_buffer)
            history_buffer = torch.cat([history_buffer[:, :, 1:, :], h_t.unsqueeze(2)], dim=2).detach()
        
        encoder_hidden_state = h_t
        
        # Decoder (RMP模式)
        outputs = []
        decoder_input = source[:, -1, :, :]
        decoder_hidden_state = encoder_hidden_state
        
        for t in range(self.horizon):
            decoder_hidden_state = self.hdyna_cell(decoder_input, decoder_hidden_state, history_buffer)
            history_buffer = torch.cat([history_buffer[:, :, 1:, :], decoder_hidden_state.unsqueeze(2)], dim=2).detach()
            
            decoder_output = self.output_layer(decoder_hidden_state)
            outputs.append(decoder_output)
            decoder_input = decoder_output
        
        return torch.stack(outputs, dim=1) 