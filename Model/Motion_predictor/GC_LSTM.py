import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class GraphConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # Dense transformations for input (constant across timesteps)
        self.w_i = nn.Linear(input_size, hidden_size)
        self.w_f = nn.Linear(input_size, hidden_size)
        self.w_o = nn.Linear(input_size, hidden_size)
        self.w_c = nn.Linear(input_size, hidden_size)

        # Graph Convolution for hidden state updates
        self.gcn_h = GraphConv(hidden_size, hidden_size)

    def forward(self, g, x, h_prev, c_prev):
        """
        g: DGL Graph (fixed topology)
        x: Fixed input vector (batch_size, input_size) (same for all timesteps)
        h_prev: Hidden state (batch_size, num_nodes, hidden_size)
        c_prev: Cell state (batch_size, num_nodes, hidden_size)
        """
        batch_size, num_nodes, _ = h_prev.shape

        # Transform input once and expand to all nodes
        x_t = self.w_i(x).unsqueeze(1).expand(-1, num_nodes, -1)  
        i_t = torch.sigmoid(x_t + self.gcn_h(g, h_prev))

        x_t = self.w_f(x).unsqueeze(1).expand(-1, num_nodes, -1)
        f_t = torch.sigmoid(x_t + self.gcn_h(g, h_prev))

        x_t = self.w_o(x).unsqueeze(1).expand(-1, num_nodes, -1)
        o_t = torch.sigmoid(x_t + self.gcn_h(g, h_prev))

        x_t = self.w_c(x).unsqueeze(1).expand(-1, num_nodes, -1)
        c_tilde = torch.tanh(x_t + self.gcn_h(g, h_prev))

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class GraphConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, seq_len=100):
        super(GraphConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm_cells = nn.ModuleList([GraphConvLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, g, x):
        """
        g: DGL Graph (fixed topology)
        x: Fixed input vector (batch_size, input_size) (same for all timesteps)
        """
        batch_size = x.shape[0]
        num_nodes = g.num_nodes()

        h = [torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(self.seq_len):  # Iterate over 100 timesteps
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](g, x, h[layer], c[layer])
            
            outputs.append(h[-1].unsqueeze(1))  # Store last layer hidden state (graph features)
        
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len=100, num_nodes, hidden_size)
        return outputs
