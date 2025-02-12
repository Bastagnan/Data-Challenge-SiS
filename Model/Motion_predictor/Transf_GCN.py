import torch
import torch.nn as nn
import torch.nn.functional as F

def adj_matrix():
    """
    Create the adjacency matrix as a PyTorch tensor for 22 joints.
    This is based on a predefined skeletal grouping.
    """
    adj_list = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20]
    ]
    
    num_nodes = max(max(sublist) for sublist in adj_list) + 1  # expected 22 joints
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for sublist in adj_list:
        for i in range(len(sublist)):
            for j in range(i + 1, len(sublist)):
                node1, node2 = sublist[i], sublist[j]
                A[node1, node2] = 1
                A[node2, node1] = 1  # Ensure symmetry

    return A


# -------------------------
# Graph Convolutional Layer
# -------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        A simple graph convolutional layer.
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
          x: Tensor of shape (B, num_nodes, in_features)
          adj: Adjacency matrix of shape (num_nodes, num_nodes)
        Returns:
          out: Tensor of shape (B, num_nodes, out_features)
        """
        # Multiply adjacency (broadcasted over the batch dimension)
        out = torch.matmul(adj, x)
        out = self.linear(out)
        return out


# -------------------------------
# Structural Refinement Module (GCN)
# -------------------------------
class StructuralRefinement(nn.Module):
    def __init__(self, in_features=3, hidden_features=64, out_features=3, num_layers=2):
        """
        Refines per-frame joint predictions by enforcing structural (skeletal) priors.
        
        Args:
          in_features (int): Input features per joint (e.g., 3 coordinates).
          hidden_features (int): Hidden dimensionality in the GCN.
          out_features (int): Output features per joint (e.g., refined 3D coordinates).
          num_layers (int): Number of GCN layers.
        """
        super(StructuralRefinement, self).__init__()
        self.input_proj = nn.Linear(in_features, hidden_features)
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(hidden_features, hidden_features) for _ in range(num_layers)]
        )
        self.relu = nn.ReLU()
        self.output_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        """
        Args:
          x: Tensor of shape (B, num_joints, in_features)
          adj: Adjacency matrix of shape (num_joints, num_joints)
        Returns:
          x: Refined tensor of shape (B, num_joints, out_features)
        """
        x = self.input_proj(x)
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.relu(x)
        x = self.output_proj(x)
        return x


# -----------------------------------------------
# Transformer Motion Predictor (Autoregressive + GCN)
# -----------------------------------------------
class TransformerMotionPredictor(nn.Module):
    """
    Transformer-based motion predictor that conditions on a text embedding 
    (e.g., from CLIP) to generate a motion sequence.
    
    The model generates motion autoregressively (using causal masking) and
    then refines each frame with a GCN to enforce skeletal priors.
    
    Args:
      embed_dim (int): Dimension of the text embedding (and model features).
      num_frames (int): Number of timesteps in the motion sequence (default: 100).
      num_joints (int): Number of joints (default: 22).
      num_layers (int): Number of Transformer decoder layers.
      num_heads (int): Number of attention heads in each decoder layer.
      dropout (float): Dropout probability.
      gcn_hidden (int): Hidden size for the structural (GCN) refinement.
      gcn_layers (int): Number of GCN layers for structural refinement.
    """
    def __init__(self, embed_dim, num_frames=100, num_joints=22, 
                 num_layers=8, num_heads=8, dropout=0.1, 
                 gcn_hidden=64, gcn_layers=4):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.motion_per_frame = num_joints * 3  # 22 joints x 3 coordinates
        
        # Learnable query tokens (one per frame) and positional embeddings.
        self.query_tokens = nn.Parameter(torch.randn(num_frames, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_frames, embed_dim))
        
        # Transformer decoder: by providing a causal mask we force autoregressive behavior.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=False  # Input shape: (seq_length, batch_size, embed_dim)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection: from transformer features to per-frame motion (flattened joints).
        self.fc_out = nn.Linear(embed_dim, self.motion_per_frame)
        
        # Structural refinement: refines each frame's joint positions using a GCN.
        self.struct_refine = StructuralRefinement(in_features=3, 
                                                  hidden_features=gcn_hidden, 
                                                  out_features=3, 
                                                  num_layers=gcn_layers)
        
        # Register the skeletal adjacency matrix as a buffer (non-trainable)
        self.register_buffer("adj", adj_matrix())

    def forward(self, text_emb):
        """
        Args:
          text_emb (Tensor): Text embedding of shape (B, embed_dim)
        Returns:
          refined_motion (Tensor): Motion tensor of shape (B, num_frames, num_joints, 3)
        """
        batch_size = text_emb.size(0)
        
        # Prepare memory for the decoder.
        # (Memory is the text embedding; unsqueeze to shape (1, B, embed_dim))
        memory = text_emb.unsqueeze(0)
        
        # Prepare autoregressive query tokens (add positional embeddings).
        queries = self.query_tokens + self.pos_embedding  # (num_frames, embed_dim)
        queries = queries.unsqueeze(1).expand(-1, batch_size, -1)  # (num_frames, B, embed_dim)
        
        # Create a causal mask so that each time step can only attend to previous ones.
        tgt_mask = torch.triu(torch.full((self.num_frames, self.num_frames), float('-inf')), diagonal=1)
        tgt_mask = tgt_mask.to(text_emb.device)
        
        # Transformer decoder (autoregressive due to tgt_mask).
        dec_out = self.decoder(tgt=queries, memory=memory, tgt_mask=tgt_mask)  # (num_frames, B, embed_dim)
        dec_out = dec_out.transpose(0, 1)  # (B, num_frames, embed_dim)
        
        # Project to motion vector (per frame: 22 joints x 3 coordinates).
        motion = self.fc_out(dec_out)  # (B, num_frames, motion_per_frame)
        motion = motion.view(batch_size, self.num_frames, self.num_joints, 3)
        
        # -------------------------------
        # Structural refinement via the GCN
        # -------------------------------
        # Process each frame independently: reshape to merge batch and time.
        motion_reshaped = motion.view(batch_size * self.num_frames, self.num_joints, 3)
        refined_motion = self.struct_refine(motion_reshaped, self.adj)
        refined_motion = refined_motion.view(batch_size, self.num_frames, self.num_joints, 3)
        
        return refined_motion.view(batch_size, self.num_frames* self.num_joints* 3)


