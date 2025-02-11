import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class GraphConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvGRUCell, self).__init__()
        self.hidden_size = hidden_size

        # Dense transformations for input (same across timesteps)
        self.w_r = nn.Linear(input_size, hidden_size)  # Reset gate
        self.w_z = nn.Linear(input_size, hidden_size)  # Update gate
        self.w_h = nn.Linear(input_size, hidden_size)  # Candidate hidden state

        # Graph Convolution for hidden state updates
        self.gcn_h = GraphConv(hidden_size, hidden_size)

    def forward(self, g, x, h_prev):
        """
        g: DGL Graph (fixed topology)
        x: Fixed input vector (batch_size, input_size) (same for all timesteps)
        h_prev: Hidden state (batch_size, num_nodes, hidden_size)
        """
        batch_size, num_nodes, _ = h_prev.shape

        # Transform input once and expand to all nodes
        x_t = self.w_r(x).unsqueeze(1).expand(-1, num_nodes, -1)  
        r_t = torch.sigmoid(x_t + self.gcn_h(g, h_prev))

        x_t = self.w_z(x).unsqueeze(1).expand(-1, num_nodes, -1)
        z_t = torch.sigmoid(x_t + self.gcn_h(g, h_prev))

        x_t = self.w_h(x).unsqueeze(1).expand(-1, num_nodes, -1)
        h_tilde = torch.tanh(x_t + r_t * self.gcn_h(g, h_prev))

        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class GraphConvGRU(nn.Module):
    def __init__(self, input_size = 32, hidden_size = 3, num_layers=1, seq_len=100):
        super(GraphConvGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.gru_cells = nn.ModuleList([GraphConvGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

        self.graph = self.build_graph()

    def build_graph(self):
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
        x: Fixed input vector (batch_size, input_size) (same for all timesteps)
        """
        batch_size = x.shape[0]
        num_nodes = g.num_nodes()

        h = [torch.zeros(batch_size, num_nodes, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(self.seq_len):  # Iterate over 100 timesteps
            for layer in range(self.num_layers):
                h[layer] = self.gru_cells[layer](self.graph, x, h[layer])
            
            outputs.append(h[-1].unsqueeze(1))  # Store last layer hidden state (graph features)
        
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len=100, num_nodes, hidden_size)
        return outputs.view(batch_size, -1)
