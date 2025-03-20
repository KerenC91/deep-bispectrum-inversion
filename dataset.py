#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:00:53 2025

@author: kerencohen2
"""
from torch.utils.data import Dataset, DataLoader
import os
import torch 
import numpy as np
from torch.utils.data.distributed import DistributedSampler




class StandardGaussianDataset(Dataset):
    
    def __init__(self, source, target):
        super().__init__()
        self.target = target
        self.source = source
        self.data_size = self.__len__()
            
        
    def __len__(self):
        return self.target.size(0)
    
    def __getitem__(self, idx):

        return idx, (self.source[idx], self.target[idx])


def read_tensor_from_matlab(file, in_train_main = False):
    if in_train_main:
        x = np.loadtxt(file, delimiter=" ")
        x = torch.tensor(x).unsqueeze(0)
    else:
        x = np.loadtxt(file, delimiter=" ")
        x = torch.tensor(x)
    return x

def read_signal(folder, k, K, label='x_true'):
    if K > 1:
        sample_path = os.path.join(folder, f'{label}_{k+1}.csv')
        target = read_tensor_from_matlab(sample_path, True)  
    else:
        sample_path = os.path.join(folder, f'{label}.csv')
        target = read_tensor_from_matlab(sample_path, True)    
    return target  


def read_sample_from_baseline(folder_read, data_size, K, N, label='x_true'):
    data_size = min(data_size, len(os.listdir(folder_read)))
    target = torch.zeros(data_size, K, N)

    print(f'The updated data size is {data_size}')

    for i in range(data_size):
        folder = os.path.join(folder_read, f'sample{i}')
        for j in range(K):
            target[i][j] = read_signal(folder, j, K, label)   
    
    return target

def read_dataset_from_baseline(folder_read, data_size, K, N, noisy):
    target = read_sample_from_baseline(folder_read, data_size, K, N)
    if noisy:
        data = read_sample_from_baseline(folder_read, data_size, K, N, label="data")
    else:
        data = target  
    
    return data, target

def generate_dataset(data_mode, data_size, K, N, noisy):
    if data_mode == 'fixed':
        # Create random dataset
        target = torch.randn(data_size, K, N)
    elif data_mode == 'random':
        # Initialize dataset to zeros and create data on the fly 
        target = torch.zeros(data_size, K, N)
    data = target
    if noisy:
        data += hparams.sigma * torch.randn(data_size, K, N)
    
    return data, target

def create_dataset(device, data_size, K, N, read_baseline, data_mode, 
                   folder_read, bs_calc, noisy=False):
    print(f'read_baseline={read_baseline}, data mode={data_mode}')
    if read_baseline: # in val dataset
        data, target = read_dataset_from_baseline(folder_read, data_size, K, N, noisy)
    else:
        data, target = generate_dataset(data_mode, data_size, K, N, noisy)
    
    source, data = bs_calc(data)        
    dataset = StandardGaussianDataset(source, target)

    return dataset


def prepare_data_loader(dataset, batch_size, is_distributed=False):
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        shuffle=False
        )
    
    return dataloader