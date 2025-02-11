import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv


class GraphConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvGRUCell, self).__init__()
        self.hidden_size = hidden_size
        # Your linear layers
        self.w_r = nn.Linear(input_size, hidden_size)
        self.w_z = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(input_size, hidden_size)

        self.gcn_h = GraphConv(hidden_size, hidden_size)
        
    def forward(self, g_batch, x, h_prev):
        """
        g_batch: batched DGL graph with (batch_size * num_nodes) total nodes.
        x: shape (batch_size, input_size).
        h_prev: shape (batch_size, num_nodes, hidden_size).
        """
        B, N, H = h_prev.shape  # e.g. (16, 22, hidden_size)
        
        # Flatten h_prev => shape (B*N, H)
        h_prev_flat = h_prev.view(B*N, H)
        
        # GraphConv(...) expects (B*N, H) for node features
        h_conv = self.gcn_h(g_batch, h_prev_flat)   # => shape (B*N, hidden_size)
        
        # We'll also flatten x for gates
        x_r = self.w_r(x)  # => shape (B, hidden_size)
        x_z = self.w_z(x)
        x_h = self.w_h(x)
        
        # Expand each to shape (B, N, H) so we can combine them with h_conv
        x_r_expanded = x_r.unsqueeze(1).expand(-1, N, -1)  # => (B, N, H)
        x_z_expanded = x_z.unsqueeze(1).expand(-1, N, -1)
        x_h_expanded = x_h.unsqueeze(1).expand(-1, N, -1)
        
        # Reshape h_conv => (B, N, H) so we can do elementwise ops
        h_conv = h_conv.view(B, N, H)
        
        # GRU gates
        r_t = torch.sigmoid(x_r_expanded + h_conv)
        z_t = torch.sigmoid(x_z_expanded + h_conv)
        h_tilde = torch.tanh(x_h_expanded + r_t * h_conv)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t


class GraphConvGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=3, num_layers=1, seq_len=100):
        super(GraphConvGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # Build base graph for 22 joints
        self.base_graph = self.build_graph()  
        
        # GRU cells
        self.gru_cells = nn.ModuleList([
            GraphConvGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def build_graph(self):
        # Exactly as you had it, returning a single DGLGraph with 22 nodes
        ...
        return g

    def forward(self, x):
        """
        x: (batch_size, input_size). This is your single "context" vector.
        """
        B = x.shape[0]
        N = self.base_graph.num_nodes()  # e.g. 22

        # Build a batched graph => repeated 'base_graph' B times
        graphs = [self.base_graph.clone() for _ in range(B)]
        g_batch = dgl.batch(graphs)  # This is now a single graph with (B*N) nodes total

        # Initialize hidden states
        h = [torch.zeros(B, N, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h[layer] = self.gru_cells[layer](g_batch, x, h[layer])
            outputs.append(h[-1].unsqueeze(1))  # Last layer hidden state

        # Cat across time => (B, seq_len, N, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        # Flatten => (B, seq_len*N*hidden_size)
        return outputs.view(B, -1)
