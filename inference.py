#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:21:17 2025

@author: kerencohen2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:46:07 2024

@author: kerencohen2
"""

import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import sys

sys.path.append(parent_dir)
sys.path.append(f'{parent_dir}/config')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from utils.utils import align_to_reference, BispectrumCalculator, compute_cost_matrix, greedy_match
from train_main import get_model, load_model_safely
from dataset import read_dataset_from_baseline, create_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config.inference_params import inference_params, inference_args
import argparse

# torch.manual_seed(234)


def evaluate(device, dataloader, baseline, model, bs_calc, args, params, output_path):

    avg_err = 0.
    
    # loop over signals
    for idx, (source, target) in dataloader:
    
        i = idx.item()
        source = source.to(device)
        target = target.to(device)
        
        # Set output folder per sample
        folder_write = os.path.join(output_path, f'sample{i+1}')
        if not os.path.exists(folder_write):
            os.mkdir(folder_write)
      
        # Pass the sample's bispectrum through the model
        output = model(source)
        
        # Compute matched loss
        cost_matrix = compute_cost_matrix(output, target, bs_calc)
        matches = greedy_match(cost_matrix)  # Get matched pairs
        
        avg_err_k = 0.
        for k, (l, j) in enumerate(matches[0], start=1):
            aligned_output, _ = align_to_reference(output[0][j], target[0][l])
            avg_err_k += cost_matrix[0, l, j].item()
            plot_comparison(i, k, folder_write, target[0][l], aligned_output)
        
        rel_error = avg_err_k / args.K
        rel_error_X_path = os.path.join(folder_write, 'rel_error_X.csv')
        np.savetxt(rel_error_X_path, [rel_error])
        print(f'sample{i}, err={rel_error}')  
        avg_err += rel_error
        print(f'curent avg={(avg_err/(i+1)):.08f}')      
    
            
    avg_err /= args.data_size
    print(f'avg err={avg_err:.08f}')   
    np.savetxt(f'{output_path}/avg_err.csv', [avg_err])     
         
        
def plot_comparison(i, k, folder_write, s_k, s_k_pred):
    folder_k = os.path.join(folder_write, f'{k}')
    if not os.path.exists(folder_k):
        os.mkdir(folder_k)
        
    # fig
    fig_path = os.path.join(folder_k, f'comp_s{k}_s{k}_pred.jpg')
    plt.figure(figsize=(9, 5))
    plt.title(f'Comparison between s{k}, s{k}_pred, sample{i + 1}')
    plt.plot(s_k.cpu().detach().numpy(), label=f's{k}', color='tab:blue')
    plt.plot(s_k_pred.cpu().detach().numpy(), label=f's{k}_pred', color='tab:orange', linestyle='dashed')
    
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(fig_path)        
    plt.close()   
        
def main(args, params, model_dir, data_dir):

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Init helpers
    bs_calc = BispectrumCalculator(args.K, args.N, device).to('cpu')
    
    if os.path.exists(data_dir):
        print(f'Reading baseline dataset...')
        read_baseline = True
    else:
        print(f'Warning: data_dir does not exist: {data_dir}, creating a new dataset.')
        read_baseline = False
        
    # Create test dataset
    dataset = create_dataset(device, 
                             args.data_size, 
                             args.K, 
                             args.N, 
                             read_baseline, 
                             args.data_mode, 
                             data_dir,
                             bs_calc,
                             args.sigma
                             )

    # Create test dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            pin_memory=False,
                            shuffle=False
                            )

    # Read baseline data
    baseline = read_dataset_from_baseline(data_dir, 
                                          args.data_size, 
                                          args.K, 
                                          args.N, 
                                          args.sigma,
                                          'x_est')
    
    # Set model path
    model_path = os.path.join(model_dir,'ckp.pt')

    # Load the model   
    model = get_model(device, args, params)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)    
    
    with torch.no_grad():
        evaluate(device, dataloader, baseline, model, bs_calc, args=args, params=params, output_path=model_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs inference on a dataset')
    parser.add_argument('--model_dir', type=str, help='directory containing a trained model (or full path to weights.pt file)')#test_folder
    parser.add_argument('--data_dir', type=str, help='directory containing validation dataset', default='')#baseline_data_folder

    args = parser.parse_args()
    main(args=inference_args, params=inference_params, model_dir=args.model_dir, data_dir=args.data_dir)