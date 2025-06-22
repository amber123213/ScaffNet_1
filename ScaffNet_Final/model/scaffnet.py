import torch
import torch.nn as nn
import torch.nn.functional as F
from model.AGCRNCell import AGCRNCell # 精确导入AGCRN的组件

class ScaffoldingGraphLearner(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(ScaffoldingGraphLearner, self).__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
    def forward(self):
        return F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

class ScaffNet(nn.Module):
    def __init__(self, args):
        super(ScaffNet, self).__init__()
        self.scaffolding_learner = ScaffoldingGraphLearner(args.num_nodes, args.embed_dim)
        # 复用AGCRN的Cell和输出层
        self.encoder = AGCRNCell(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim)
        self.end_conv = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.hidden_dim = args.rnn_units
        self.num_nodes = args.num_nodes

    def forward(self, source, teacher_forcing_ratio=0.5):
        # source: B, T, N, D
        batch_size = source.shape[0]
        seq_len = source.shape[1]
        device = source.device
        
        # 创建初始隐藏状态
        init_state = self.encoder.init_hidden_state(batch_size).to(device)
        
        # 手动处理序列并收集隐藏状态历史
        hidden_history = []
        current_state = init_state
        
        # 直接获取节点嵌入
        node_embeddings = self.scaffolding_learner.node_embeddings
        
        for t in range(seq_len):
            # 获取当前时间步的输入
            current_input = source[:, t, :, :]
            # 通过AGCRNCell处理当前时间步
            current_state = self.encoder(current_input, current_state, node_embeddings)
            # 保存当前隐藏状态
            hidden_history.append(current_state)
        
        # 使用最后一个隐藏状态进行预测
        output = current_state.unsqueeze(1)  # B, 1, N, hidden
        
        # 应用卷积层进行预测
        output = self.end_conv(output)  # B, T*C, N, 1
        
        if self.training:
            A_scaffold = self.scaffolding_learner()
            return output, torch.stack(hidden_history, dim=1), A_scaffold
        else:
            return output 