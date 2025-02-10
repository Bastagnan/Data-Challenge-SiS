import numpy as np
import random
import os
from os.path import join as pjoin
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from  dataset_dataloader import MotionDataset

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
        text_emb = self.text_encoder(text_tokens)
        motion_pred = self.motion_predictor(text_emb)
        return motion_pred


# --------------------
#  TRAINING LOOP
# --------------------
def train( model, data_dir='./path/to/data', num_epochs=10, batch_size=16, lr=1e-3,
          max_seq_len=64, vocab_size=128, embed_dim=32, n_frames=100, n_joints=22):
    """
    Example training function that uses the MotionDataset and a simple pipeline.
    Modify as needed.
    """
    # 1) Build dataset & dataloader
    train_set = MotionDataset(data_dir, 'train.txt', mean=None, std=None)
    valid_set = MotionDataset(data_dir, 'val.txt', mean=None, std=None)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # 2) Model & optimizer
    motion_dim = n_frames * n_joints * 3  # Flatten: (T, J, 3)
    model = Text2MotionPipeline(Text_Encoder, Motion_predictor, vocab_size, embed_dim, motion_dim)
    model.cuda()  # if you have a GPU

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3) Training epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for motions, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # motions: shape (batch_size, n_frames, n_joints, 3) in your dataset
            # texts: list of strings
            motions = motions.cuda()  # (batch_size, T, J, 3)
            
            # Flatten ground-truth motions to (batch_size, T*J*3)
            gt_motion = motions.view(motions.size(0), -1)

            # Forward pass
            pred_motion = model(text_tokens)  # (batch_size, T*J*3)
            
            # Compute loss
            loss = criterion(pred_motion, gt_motion)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(motions)
        
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        # 4) Validation step (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for motions, texts in valid_loader:
                motions = motions.cuda()
                gt_motion = motions.view(motions.size(0), -1)
                
                pred_motion = model(text)
                loss = criterion(pred_motion, gt_motion)
                val_loss += loss.item() * len(motions)
            
            val_loss /= len(valid_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}")

    print("Training complete!")


if __name__ == '__main__':

    data_dir = './'
    train_set = MotionDataset(data_dir, 'train.txt', mean=None, std=None)
    valid_set = MotionDataset(data_dir, 'val.txt', mean=None, std=None)

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)


    for motion, text in train_loader:
        print('motion shape:', motion.shape)
        print('exemple of texts:', text[0])
        break
    
    # train(
    #     model,
    #     data_dir='./path/to/data', 
    #     num_epochs=10, 
    #     batch_size=16, 
    #     lr=1e-3,
    #     max_seq_len=64,
    #     vocab_size=128,
    #     embed_dim=32,
    #     n_frames=100,  # for example
    #     n_joints=22    # for example
    # )
