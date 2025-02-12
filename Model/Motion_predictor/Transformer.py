import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerMotionPredictor(nn.Module):
    """
    Transformer-based motion predictor that conditions on a text embedding 
    (e.g., from CLIP) to generate a motion sequence of shape (num_frames, num_joints, 3).
    
    Args:
      embed_dim (int): Dimension of the text embedding (and model features).
      num_frames (int): Number of timesteps in the motion sequence (default: 100).
      num_joints (int): Number of joints (default: 22).
      num_layers (int): Number of Transformer decoder layers.
      num_heads (int): Number of attention heads in each decoder layer.
      dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim, num_frames=100, num_joints=22, 
                 num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.motion_per_frame = num_joints * 3  # Each frame outputs 22 joints x 3 coords = 66
        
        # Learnable query tokens for each frame.
        # These queries will be updated by the transformer decoder.
        self.query_tokens = nn.Parameter(torch.randn(num_frames, embed_dim))
        
        # Learnable positional embeddings (can also use sinusoidal embeddings)
        self.pos_embedding = nn.Parameter(torch.randn(num_frames, embed_dim))
        
        # Transformer decoder: note that PyTorch's nn.TransformerDecoder expects inputs
        # with shape (target_seq_length, batch_size, embed_dim).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=False  # Default is False: shape is (S, B, E)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection: from the transformer output to the per-frame motion vector.
        self.fc_out = nn.Linear(embed_dim, self.motion_per_frame)
        
    def forward(self, text_emb):
        """
        Forward pass.
        
        Args:
          text_emb (Tensor): Text embedding tensor of shape (B, embed_dim),
                             e.g. produced by the CLIP text encoder.
                             
        Returns:
          motion (Tensor): Generated motion of shape (B, num_frames, num_joints, 3)
        """
        batch_size = text_emb.size(0)
        
        # Prepare the memory for the decoder.
        # Since text_emb is one vector per sample, we unsqueeze to get (1, B, embed_dim).
        memory = text_emb.unsqueeze(0)  # Shape: (S=1, B, embed_dim)
        
        # Prepare query tokens for each frame:
        # Add learned positional embeddings to the query tokens.
        queries = self.query_tokens + self.pos_embedding  # Shape: (num_frames, embed_dim)
        # Expand queries for the batch: (num_frames, B, embed_dim)
        queries = queries.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Decode using the transformer decoder.
        # Note: tgt=queries has shape (num_frames, B, embed_dim) and memory is (1, B, embed_dim)
        dec_out = self.decoder(tgt=queries, memory=memory)  # Output: (num_frames, B, embed_dim)
        
        # Transpose to shape (B, num_frames, embed_dim)
        dec_out = dec_out.transpose(0, 1)
        
        # Project each timestep representation to a motion vector (22*3=66 values)
        motion = self.fc_out(dec_out)  # Shape: (B, num_frames, motion_per_frame)
        
        # Reshape to (B, num_frames, num_joints, 3)
        motion = motion.view(batch_size, self.num_frames, self.num_joints, 3)
        return motion



