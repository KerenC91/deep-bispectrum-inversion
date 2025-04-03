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


def read_signal(folder, k, K, label='x_true'):
    if K > 1:
        sample_path = os.path.join(folder, f'{label}_{k+1}.csv')
    else:
        sample_path = os.path.join(folder, f'{label}.csv')
    target = np.loadtxt(sample_path, delimiter=" ")
    target = torch.tensor(target)
    
    return target


def read_bispectrum(folder, L):
    sample_path = os.path.join(folder, 'b_mixed.csv')
    source = np.loadtxt(sample_path, delimiter=",")
    source = np.reshape(source, (2, L, L), order='F')
    source = torch.tensor(source)

    return source

def read_samples_from_baseline(folder_read, data_size, K, L, label='x_true'):

    data_size = min(data_size, len(os.listdir(folder_read)))
    target = torch.zeros(data_size, K, L)
    source = torch.zeros(data_size, 2, L, L)
    print(f'The updated data size is {data_size}')

    for i in range(data_size):
        sample_path = os.path.join(folder_read, f'sample{i}')
        
        # Read the bispectrum
        bs = read_bispectrum(sample_path, L)
        source[i] = bs
        # Read the underlying targets
        for k in range(K):
            signal = read_signal(sample_path, k, K, label)
            target[i][k] = signal

    return source, target

def read_dataset_from_baseline(folder_read, data_size, K, L, sigma, label="x_true"):
    source, target = read_samples_from_baseline(folder_read, data_size, K, L, label)

    return source, target

def generate_dataset(data_mode, data_size, K, L, sigma):
    if data_mode == 'fixed':
        # Create random dataset
        target = torch.randn(data_size, K, L)
    elif data_mode == 'random':
        # Initialize dataset to zeros and create data on the fly 
        target = torch.zeros(data_size, K, L)
    data = target
    if sigma:
        data += sigma * torch.randn(data_size, K, L)
    
    return data, target


def create_dataset(data_size, K, L, read_baseline, data_mode,
                   folder_read, bs_calc, sigma=0., label="x_true"):
    print(f'read_baseline={read_baseline}, data mode={data_mode}')
    if read_baseline: # in val dataset
        source, target = read_dataset_from_baseline(folder_read, data_size, K, L, sigma, label)
    else:
        data, target = generate_dataset(data_mode, data_size, K, L, sigma)
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
