import torch
import numpy as np
import pandas as pd
from os.path import join as pjoin
from torch.utils.data import DataLoader

from dataset_dataloader import TestDataset

def predict_and_submit(
    model,
    data_dir="/kaggle/input/motion",
    test_file="test.txt",
    output_csv="submission.csv",
    batch_size=16,
    n_frames=100,
    n_joints=22
):
    """
    1) Loads the test dataset (with text).
    2) Runs inference with the trained model.
    3) Flattens predicted motions to (N, 6600).
    4) Writes them to a CSV with columns [id, f_0, ..., f_6599].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to eval mode & GPU/CPU
    model.eval()
    model = model.to(device)
    
    # Create test dataset / loader
    #  => Adjust mean/std or other dataset arguments as needed
    test_set = TestDataset(data_dir, test_file)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    
    with torch.no_grad():
        for texts in test_loader:
            
            pred_motions = model(texts)  # shape: (batch_size, 6600)
            # Move to CPU, convert to NumPy
            pred_motions = pred_motions.cpu().numpy()  # shape: (B, 6600)
            
            # We'll accumulate predictions
            all_preds.append(pred_motions)
    
    # Concatenate all predictions: shape => (N, 6600)
    all_preds = np.concatenate(all_preds, axis=0)
    
    ## read ids
    with open(pjoin(data_dir, test_file)) as fd:
        test_motion_ids = fd.read().strip().split('\n')
    
    # Build the submission rows
    submission_data = []
    for motion_id, flattened_motion in zip(test_motion_ids, all_preds):
        submission_data.append([motion_id] + flattened_motion.tolist())
    
    # Construct column names
    num_feats = n_frames * n_joints * 3  # 6600 if 100x22x3
    columns = ["id"] + [f"f_{i}" for i in range(num_feats)]
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_data, columns=columns)
    
    # Save to CSV
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved submission to: {output_csv}")
    
    return submission_df


def predict(
    model,
    data_dir="/kaggle/input/motion",
    test_file="test.txt",
    batch_size=16,
    n_frames=100,
    n_joints=22
):
    """
    1) Loads the test dataset (with text).
    2) Runs inference with the trained model.
    3) Flattens predicted motions to (N, 6600).
    4) Writes them to a CSV with columns [id, f_0, ..., f_6599].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to eval mode & GPU/CPU
    model.eval()
    model = model.to(device)
    
    # Create test dataset / loader
    #  => Adjust mean/std or other dataset arguments as needed
    test_set = TestDataset(data_dir, test_file)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_texts = []
    
    with torch.no_grad():
        for texts in test_loader:
            
            pred_motions = model(texts)  # shape: (batch_size, 6600)
            # Move to CPU, convert to NumPy
            pred_motions = pred_motions.cpu().numpy()  # shape: (B, 6600)
            
            # We'll accumulate predictions
            all_preds.append(pred_motions.reshape(batch_size,100,22,3))
            all_texts.append(texts)
    
    # Concatenate all predictions: shape => (N, 100,22,3)
    all_preds = np.concatenate(all_preds, axis=0)

    with open(pjoin(data_dir, test_file)) as fd:
        test_motion_ids = fd.read().strip().split('\n')
    
    return test_motion_ids, all_texts, all_preds
