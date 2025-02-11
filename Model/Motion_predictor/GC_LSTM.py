import torch
import torch.nn as nn
import dgl
import numpy as np
from dgl.nn import GraphConv


class GraphConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # Dense transformations for input
        self.w_i = nn.Linear(input_size, hidden_size)
        self.w_f = nn.Linear(input_size, hidden_size)
        self.w_o = nn.Linear(input_size, hidden_size)
        self.w_c = nn.Linear(input_size, hidden_size)

        # Graph Convolution for hidden state
        self.gcn_h = GraphConv(hidden_size, hidden_size)

    def forward(self, g_batch, x, h_prev, c_prev):
        """
        g_batch: DGL batched graph with (batch_size * num_nodes) total nodes
        x: (batch_size, input_size)
        h_prev: (batch_size, num_nodes, hidden_size)
        c_prev: (batch_size, num_nodes, hidden_size)
        """
        B, N, H = h_prev.shape  # e.g. (16, 22, hidden_size)

        # Flatten h_prev => shape (B*N, H)
        h_prev_flat = h_prev.view(B * N, H)

        # GraphConv expects (B*N, H)
        h_conv = self.gcn_h(g_batch, h_prev_flat)  # => shape (B*N, H)
        h_conv = h_conv.view(B, N, H)  # reshape back to (B, N, H)

        # i/f/o/c gates need transformations of x
        x_i = self.w_i(x)  # => (B, H)
        x_f = self.w_f(x)
        x_o = self.w_o(x)
        x_c = self.w_c(x)

        # Expand each to (B, N, H) for node-wise ops
        x_i_expanded = x_i.unsqueeze(1).expand(-1, N, -1)
        x_f_expanded = x_f.unsqueeze(1).expand(-1, N, -1)
        x_o_expanded = x_o.unsqueeze(1).expand(-1, N, -1)
        x_c_expanded = x_c.unsqueeze(1).expand(-1, N, -1)

        i_t = torch.sigmoid(x_i_expanded + h_conv)
        f_t = torch.sigmoid(x_f_expanded + h_conv)
        o_t = torch.sigmoid(x_o_expanded + h_conv)
        c_tilde = torch.tanh(x_c_expanded + h_conv)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class GraphConvLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=3, num_layers=1, seq_len=100):
        super(GraphConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.lstm_cells = nn.ModuleList([
            GraphConvLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])

        # Build a "base graph" once with 22 nodes
        self.base_graph = self.build_graph().to('cuda') 
        print('to Cuda')

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
        x: shape (batch_size, input_size)
        """
        B = x.shape[0]
        N = self.base_graph.num_nodes()

        # Build a batched graph: replicate base_graph B times
        graphs = [self.base_graph.clone() for _ in range(B)]
        g_batch = dgl.batch(graphs)  # single graph with B*N nodes

        # Initialize hidden states
        h = [
            torch.zeros(B, N, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(B, N, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](
                    g_batch, x, h[layer], c[layer]
                )
            outputs.append(h[-1].unsqueeze(1))  # last layer hidden state

        print(len(outputs), outputs[0].size())

        # (B, seq_len, N, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        # flatten => (B, seq_len*N*hidden_size)
        print(f"Raw outputs shape: {outputs.shape}")

        outputs = outputs.view(B, -1)  # Should be (B, 6600)

        print(f"Final output shape: {outputs.shape}")
        return outputs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphConvLSTM(input_size=32, hidden_size=3, num_layers=1, seq_len=100).to(device)

    B = 4  # small batch for testing
    x = torch.randn(B, 32, device=device)   # shape (batch_size, input_size)
    out = model(x)
    print("Output shape:", out.shape)
    # Expected => (4, 100*22*3) = (4, 6600)

