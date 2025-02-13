import numpy as np
import random
import os
from os.path import join as pjoin
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Import your MotionDataset, CLIPTextEncoder, MLP from the same directory/package
from dataset_dataloader import MotionDataset
from Text_encoder.CLIP import CLIPTextEncoder
from Motion_predictor.MLP import MLP

# --------------------
#  WRAPPER MODEL
# --------------------
class Text2MotionPipeline(nn.Module):
    """
    Pipeline that first encodes text (using CLIP) and then generates motion
    using a Transformer-based motion predictor.
    """
    def __init__(self, Text_Encoder, Motion_predictor, vocab_size=128, embed_dim=32, motion_dim=6600):
        super().__init__()
        self.text_encoder = Text_Encoder(vocab_size, embed_dim)
        self.motion_predictor = Motion_predictor(embed_dim)
    
    def forward(self, text_tokens):
        # text_tokens: list of strings.
        # Get text embedding from CLIP encoder.
        print("bbbbbbbb")
        text_emb = self.text_encoder(text_tokens)  # Expected shape: (B, embed_dim)
        # Generate motion from text embedding.
        motion_pred = self.motion_predictor(text_emb)  # Shape: (B, 100, 22, 3)
        # Depending on your training loop, you might flatten this to (B, motion_dim)
        # For example: motion_pred = motion_pred.view(motion_pred.size(0), -1)
        return motion_pred


def adj_matrix():
    """
    Create the adjacency matrix as a PyTorch tensor.
    
    Returns:
    - A: Tensor of shape (N, N) representing the adjacency matrix.
    """
    adj_list = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20]
    ]
    
    num_nodes = max(max(sublist) for sublist in adj_list) + 1
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for sublist in adj_list:
        for i in range(len(sublist)):
            for j in range(i + 1, len(sublist)):
                node1, node2 = sublist[i], sublist[j]
                A[node1, node2] = 1
                A[node2, node1] = 1  # Ensure symmetry

    return A


def compute_initial_distances_torch(X0, A):
    """
    Compute d_{ij}^0 for each edge in the graph based on the initial positions.
    
    Args:
    - X0: Tensor of shape (B, N, D), initial node positions for the batch.
    - A: Tensor of shape (N, N), adjacency matrix.
    
    Returns:
    - d0_matrix: Tensor of shape (B, N, N) with initial edge distances.
    """
    indices = torch.where(A > 0)  # Get indices of connected nodes
    d0_matrix = torch.zeros_like(A, dtype=torch.float32).unsqueeze(0).expand(X0.shape[0], -1, -1)  # (B, N, N)
    
    distances = torch.norm(X0[:, indices[0], :] - X0[:, indices[1], :], dim=-1)  # (B, num_edges)
    d0_matrix[:, indices[0], indices[1]] = distances
    d0_matrix[:, indices[1], indices[0]] = distances  # Symmetric
    
    return d0_matrix

def loss_distance_between_points_torch(X_gt, X_seq, A):
    """
    Compute the batch loss ensuring that each edge maintains its initial distance.
    
    Args:
    - X_seq: Tensor of shape (B, T, N, D), node positions over time for a batch.
    - A: Tensor of shape (N, N), adjacency matrix.
    
    Returns:
    - loss: Scalar tensor (mean squared loss).
    """
    B, T, N, D = X_seq.shape
    d0_matrix = compute_initial_distances_torch(X_gt[:, 0], A)  # (B, N, N)
    loss = torch.tensor(0.0, device=X_seq.device)

    for t in range(T):
        indices = torch.where(A > 0)  # Get connected node indices
        distances = torch.norm(X_seq[:, t, indices[0], :] - X_seq[:, t, indices[1], :], dim=-1)  # (B, num_edges)
        loss += torch.sum((distances - d0_matrix[:, indices[0], indices[1]]) ** 2)

    return loss / (B * T)  # Normalize by batch size and time steps



# --------------------
#  TRAINING LOOP
# --------------------
def train(model, 
          data_dir='/kaggle/input/motion',  # point to Kaggle dataset
          train_loader = None,
          val_loader = None,
          num_epochs=10, 
          batch_size=16, 
          lr=1e-3,
          n_frames=100, 
          n_joints=22):

    if train_loader == None and val_loader == None:
        # Create dataset / dataloaders
        train_set = MotionDataset(data_dir, 'train.txt', mean=None, std=None)
        valid_set = MotionDataset(data_dir, 'val.txt', mean=None, std=None)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # On GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Force the model to use float32 (fixes any mismatch with half precision)
    model = model.float()

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Flatten motion size
    motion_dim = n_frames * n_joints * 3

    A = adj_matrix().to(device) 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for motions, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # motions: (batch_size, n_frames, n_joints, 3)
            # texts: list of strings

            motions = motions.to(device)
            # Also ensure motions are float32
            motions = motions.float()

            # Flatten GT motion
            gt_motion = motions.view(motions.size(0), -1)  # (batch_size, motion_dim)

            # Forward pass with CLIP encoder -> motion predictor
            pred_motion = model(texts)  # shape => (batch_size, motion_dim) in float32

            # print(torch.reshape(gt_motion, (gt_motion.size(0), 100, 22, 3)).shape, pred_motion.size())

            # distance_loss = loss_distance_between_points_torch(torch.reshape(gt_motion, (gt_motion.size(0), 100, 22, 3)) ,torch.reshape(pred_motion, (gt_motion.size(0), 100, 22, 3)), A)

            loss = criterion(pred_motion, gt_motion) 
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * len(motions)

        scheduler.step()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for motions, texts in val_loader:
                motions = motions.to(device).float()
                gt_motion = motions.view(motions.size(0), -1)
                pred_motion = model(texts)
                loss = criterion(pred_motion, gt_motion)
                val_loss += loss.item() * len(motions)

        val_loss /= len(val_loader.dataset)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

    print("Training complete!")



if __name__ == '__main__':
    # Suppose your motion is shape (100 frames, 22 joints, 3 coords) => motion_dim=6600
    motion_dim = 100 * 22 * 3

    # Build the pipeline
    model = Text2MotionPipeline(
        Text_Encoder=CLIPTextEncoder,
        Motion_predictor=MLP,
        vocab_size=128,  # not used by CLIP, but must be passed
        embed_dim=32,    # your chosen dimension (CLIP gets projected to 32)
        motion_dim=motion_dim
    )

    # Train
    train(
        model=model,
        data_dir='/kaggle/input/motion',  # make sure Kaggle dataset has train.txt/val.txt
        num_epochs=10,
        batch_size=16,
        lr=1e-3,
        n_frames=100,
        n_joints=22
    )
