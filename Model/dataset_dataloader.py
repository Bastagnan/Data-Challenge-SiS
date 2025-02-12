import numpy as np
import random
import os
from os.path import join as pjoin
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class MotionDataset(Dataset):
    
    def __init__(self, data_dir, ids_file, mean=None, std=None):
        
        self.data_dir = data_dir
        self.ids_file  = ids_file
        self.mean = mean
        self.std = std
        
        ## read ids
        with open(pjoin(data_dir, ids_file)) as fd:
            self.list_ids = fd.read().strip().split('\n')
            
        ## load data
        motions, texts = [], []
        for file_id in tqdm(self.list_ids, desc='loading data...'):
            ## get paths
            motion_path = pjoin(self.data_dir, 'motions', file_id + '.npy')
            text_path = pjoin(self.data_dir, 'texts', file_id + '.txt')
            
            ## load motion
            motion = np.load(motion_path)
            
            ## load text
            with open(text_path) as fd:
                motion_descriptions = fd.read().strip().split('\n')
            
            motions.append(motion)
            texts.append(motion_descriptions)
            
        self.motions = np.array(motions)
        self.texts = texts
        
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        motion = self.motions[index]
        motion_texts = self.texts[index]
        
        ## pick random text
        # text = random.choice(motion_texts)
        # text = text.split('#')[0]
        
        ## normalize motion
        if self.mean is not None and self.std is not None:
            motion = (motion - self.mean) / self.std
            
        motion = torch.from_numpy(motion)
        
        return motion, text
    
class TestDataset(Dataset):
    
    def __init__(self, data_dir, ids_file):
        
        self.data_dir = data_dir
        self.ids_file  = ids_file
        
        ## read ids
        with open(pjoin(data_dir, ids_file)) as fd:
            self.list_ids = fd.read().strip().split('\n')
            
        ## load data
        texts = []
        for file_id in tqdm(self.list_ids, desc='loading data...'):
            ## get paths
            text_path = pjoin(self.data_dir, 'texts', file_id + '.txt')
            
            ## load text
            with open(text_path) as fd:
                motion_descriptions = fd.read().strip().split('\n')
            
            texts.append(motion_descriptions)
            
        self.texts = texts
        
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        motion_texts = self.texts[index]
        
        ## pick random text
        text = random.choice(motion_texts)
        text = text.split('#')[0]
        
        return text


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