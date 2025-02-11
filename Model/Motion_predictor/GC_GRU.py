import torch
import torch.nn as nn
import dgl
import numpy as np
from dgl.nn import GraphConv


class GraphConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Linear layers for GRU gates
        self.w_r = nn.Linear(input_size, hidden_size)
        self.w_z = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(input_size, hidden_size)

        # Graph Convolution for hidden state
        self.gcn_h = GraphConv(hidden_size, hidden_size)
        
    def forward(self, g_batch, x, h_prev):
        """
        g_batch: batched DGL graph with (batch_size * num_nodes) total nodes.
        x: shape (batch_size, input_size).
        h_prev: shape (batch_size, num_nodes, hidden_size).
        """
        B, N, H = h_prev.shape  
        
        # Flatten h_prev for GraphConv
        h_prev_flat = h_prev.view(B * N, H)
        
        # GraphConv operation
        h_conv = self.gcn_h(g_batch, h_prev_flat)
        h_conv = h_conv.view(B, N, H)
        
        # Compute GRU gates
        x_r = self.w_r(x).unsqueeze(1).expand(-1, N, -1)
        x_z = self.w_z(x).unsqueeze(1).expand(-1, N, -1)
        x_h = self.w_h(x).unsqueeze(1).expand(-1, N, -1)
        
        r_t = torch.sigmoid(x_r + h_conv)
        z_t = torch.sigmoid(x_z + h_conv)
        h_tilde = torch.tanh(x_h + r_t * h_conv)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


class GraphConvGRU(nn.Module):
    def __init__(self, input_size=128, hidden_size=3, num_layers=2, seq_len=100):
        super(GraphConvGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # GRU cells
        self.gru_cells = nn.ModuleList([
            GraphConvGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        # Base graph (22 nodes)
        self.base_graph = self.build_graph().to('cuda')

    def build_graph(self):
        # Same adjacency building as before
        adj_list = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20]
        ]
        num_nodes = max(max(sublist) for sublist in adj_list) + 1
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for sublist in adj_list:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    node1, node2 = sublist[i], sublist[j]
                    adj_matrix[node1, node2] = 1
                    adj_matrix[node2, node1] = 1

        src, dst = np.nonzero(adj_matrix)
        g = dgl.graph((src, dst))
        
        return g

    def forward(self, x):
        """
        x: (batch_size, input_size)
        """
        B = x.shape[0]
        N = self.base_graph.num_nodes()
        
        # Batch graph replication
        graphs = [self.base_graph.clone() for _ in range(B)]
        g_batch = dgl.batch(graphs)  # Single graph with (B*N) nodes

        # Initialize hidden states
        h = [torch.zeros(B, N, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h[layer] = self.gru_cells[layer](g_batch, x if layer == 0 else h[layer-1], h[layer])
            outputs.append(h[-1].view(B, N, self.hidden_size).unsqueeze(1))

        # (B, seq_len, N, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        # Flatten => (B, seq_len*N*hidden_size)
        return outputs.view(B, -1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphConvGRU(input_size=32, hidden_size=3, num_layers=1, seq_len=100).to(device)
    
    B = 4  # Batch size
    x = torch.randn(B, 32, device=device)  # shape (batch_size, input_size)
    out = model(x)
    print("Output shape:", out.shape)  # Expected: (B, 100*22*3) = (4, 6600)
