import torch
import torch.nn as nn
from model.hdyna_cell import H_DYNA_Cell

class H_DYNA_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, node_embed_dim, M_global, S_global, M_local, S_local, pattern_dim, num_layers=1):
        super(H_DYNA_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one H-DYNA layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.hdyna_cells = nn.ModuleList()
        self.hdyna_cells.append(H_DYNA_Cell(node_num, dim_in, dim_out, node_embed_dim, M_global, S_global, M_local, S_local, pattern_dim))
        for _ in range(1, num_layers):
            self.hdyna_cells.append(H_DYNA_Cell(node_num, dim_out, dim_out, node_embed_dim, M_global, S_global, M_local, S_local, pattern_dim))

    def forward(self, x, init_state):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.hdyna_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(torch.zeros(batch_size, self.node_num, self.hdyna_cells[i].hidden_dim))
        return torch.stack(init_states, dim=0)

class H_DYNA(nn.Module):
    def __init__(self, args):
        super(H_DYNA, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        # H-DYNA specific parameters
        self.node_embed_dim = args.embed_dim
        self.M_global = args.M_global
        self.S_global = args.S_global
        self.M_local = args.M_local
        self.S_local = args.S_local
        self.pattern_dim = args.pattern_dim

        self.encoder = H_DYNA_Encoder(args.num_nodes, args.input_dim, args.rnn_units,
                                    self.node_embed_dim, self.M_global, self.S_global,
                                    self.M_local, self.S_local, self.pattern_dim,
                                    args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output