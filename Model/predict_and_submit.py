import torch
import numpy as np
import pandas as pd
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_dataloader import TestDataset

def predict_and_submit(
    model,
    data_dir="/kaggle/input/motion",
    test_file="test.txt",
    output_csv="submission.csv",
    batch_size=16,
    n_frames=100,
    n_joints=22,
    mean=None,
    std=None
):
    """
    1) Loads the test dataset (with text).
    2) Runs inference with the trained model.
    3) Denormalizes predictions using the provided mean and std.
    4) Flattens predicted motions to (N, 6600).
    5) Writes them to a CSV with columns [id, f_0, ..., f_6599].
    
    Args:
      model: The trained model.
      data_dir (str): Base directory of the data.
      test_file (str): File containing test IDs.
      output_csv (str): Output CSV filename.
      batch_size (int): Batch size for inference.
      n_frames (int): Number of frames per motion.
      n_joints (int): Number of joints per frame.
      mean (float or np.ndarray, optional): Mean used for normalization.
      std (float or np.ndarray, optional): Standard deviation used for normalization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set model to evaluation mode and move to device.
    model.eval()
    model = model.to(device)
    
    # Create the test dataset and loader.
    test_set = TestDataset(data_dir, test_file)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    
    with torch.no_grad():
        for texts in tqdm(test_loader, desc='predict...'):
            pred_motions = model(texts)  # shape: (batch_size, 6600)
            
            # Denormalize the predictions if mean and std are provided.
            if mean is not None and std is not None:
                pred_motions = pred_motions * std + mean
            
            # Move to CPU and convert to NumPy.
            pred_motions = pred_motions.cpu().numpy()  # shape: (B, 6600)
            
            all_preds.append(pred_motions)
    
    # Concatenate all predictions: shape => (N, 6600)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Read the test IDs.
    with open(pjoin(data_dir, test_file)) as fd:
        test_motion_ids = fd.read().strip().split('\n')
    
    # Build the submission rows.
    submission_data = []
    for motion_id, flattened_motion in zip(test_motion_ids, all_preds):
        submission_data.append([motion_id] + flattened_motion.tolist())
    
    # Construct column names.
    num_feats = n_frames * n_joints * 3  # 6600 if 100 x 22 x 3.
    columns = ["id"] + [f"f_{i}" for i in range(num_feats)]
    
    # Create a DataFrame and save to CSV.
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved submission to: {output_csv}")
    
    return submission_df


def predict(
    model,
    data_dir="/kaggle/input/motion",
    test_file="test.txt",
    batch_size=16,
    n_frames=100,
    n_joints=22,
    mean=None,
    std=None
):
    """
    1) Loads the test dataset (with text).
    2) Runs inference with the trained model.
    3) Denormalizes predictions using the provided mean and std.
    4) Reshapes predicted motions to (N, n_frames, n_joints, 3).
    5) Returns the test IDs, texts, and predicted motions.
    
    Args:
      model: The trained model.
      data_dir (str): Base directory of the data.
      test_file (str): File containing test IDs.
      batch_size (int): Batch size for inference.
      n_frames (int): Number of frames per motion.
      n_joints (int): Number of joints per frame.
      mean (float or np.ndarray, optional): Mean used for normalization.
      std (float or np.ndarray, optional): Standard deviation used for normalization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set model to evaluation mode and move to device.
    model.eval()
    model = model.to(device)
    
    # Create the test dataset and loader.
    test_set = TestDataset(data_dir, test_file)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_texts = []
    
    with torch.no_grad():
        for texts in tqdm(test_loader, desc='predict...'):
            pred_motions = model(texts)  # shape: (batch_size, 6600)
            
            # Denormalize the predictions if mean and std are provided.
            if mean is not None and std is not None:
                pred_motions = pred_motions * std + mean
            
            # Move to CPU and convert to NumPy.
            pred_motions = pred_motions.cpu().numpy()  # shape: (B, 6600)
            B = pred_motions.shape[0]
            
            # Reshape predictions to (B, n_frames, n_joints, 3).
            all_preds.append(pred_motions.reshape(B, n_frames, n_joints, 3))
            all_texts.append(texts)
    
    # Concatenate all predictions.
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Read the test IDs.
    with open(pjoin(data_dir, test_file)) as fd:
        test_motion_ids = fd.read().strip().split('\n')
    
    return test_motion_ids, all_texts, all_preds
