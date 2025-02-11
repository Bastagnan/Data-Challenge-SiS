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
    Encapsulate both:
      1) a text encoder
      2) a motion predictor (embedding -> motion)
    """
    def __init__(self, Text_Encoder, Motion_predictor, vocab_size=128, embed_dim=32, motion_dim=6600):
        super().__init__()
        self.text_encoder = Text_Encoder(vocab_size, embed_dim)
        self.motion_predictor = Motion_predictor(embed_dim, motion_dim)

    def forward(self, text_tokens):
        # text_tokens is a list of strings
        # We'll ensure that everything within text_encoder returns float32
        text_emb = self.text_encoder(text_tokens)
        # text_emb should be float32
        motion_pred = self.motion_predictor(text_emb)
        return motion_pred

# --------------------
#  TRAINING LOOP
# --------------------
def train(model, 
          data_dir='/kaggle/input/motion',  # point to Kaggle dataset
          train_dataloader = None,
          val_dataloader = None,
          num_epochs=10, 
          batch_size=16, 
          lr=1e-3,
          n_frames=100, 
          n_joints=22):

    if train_dataloader == None and val_dataloader == None:
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

    # Flatten motion size
    motion_dim = n_frames * n_joints * 3

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

            loss = criterion(pred_motion, gt_motion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(motions)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for motions, texts in valid_loader:
                motions = motions.to(device).float()
                gt_motion = motions.view(motions.size(0), -1)
                pred_motion = model(texts)
                loss = criterion(pred_motion, gt_motion)
                val_loss += loss.item() * len(motions)

        val_loss /= len(valid_loader.dataset)
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
