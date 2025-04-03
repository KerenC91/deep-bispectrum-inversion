#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:49:44 2024

@author: kerencohen2
"""
import torch
import argparse
from torch.cuda import device_count
from config.params import params
from train_main import train, train_distributed
import os



def is_torchrun():
    if 'LOCAL_RANK' in os.environ:
        return True
    return False

def main(args):
    replica_count = device_count() if is_torchrun() else 1
    
    if args.L % args.window_size != 0:
        raise ValueError(f'Signal size {args.L} is not evenly divisble by window size {args.window_size}. ' 
                         f'Please choose a suitable window size.')
    if replica_count > 1:
        if args.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        args.batch_size = args.batch_size // replica_count
        train_distributed(args, params)
    else:
        if torch.cuda.is_available():
            print("Running with a single GPU")
        else:
            print("GPU is not available, running with CPU")
        train(args, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Bispectrum Inversion.')
    
    # Core settings
    parser.add_argument('--L', type=int, default=10, help='Signal length in the dataset.')
    parser.add_argument('--K', type=int, default=1, help='Number of signals to reconstruct from.')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-Batch size.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate. If using a scheduler: '
        'maximal for OneCycleLR and CyclicLR, else initial.')
    parser.add_argument('--optimizer', type=str, default="AdamW", 
        help='Optimizer: Adam, SGD, RMSprop, or AdamW. Update relevant parameters in the configuration.')
    parser.add_argument('--scheduler', type=str, default='None',
        help='Learning rate scheduler: StepLR, ReduceLROnPlateau, OneCycleLR, '
             'CosineAnnealingLR, CyclicLR, Manual, or None (no scheduler). '
    	 'Update relevant parameters in the configuration.')
    
    # Data
    parser.add_argument('--data_mode', type=str, default='random', help='Training data mode: random or fixed.')
    parser.add_argument('--train_data_size', type=int, default=5000, help='Training set size.')
    parser.add_argument('--val_data_size', type=int, default=100, help='Validation set size.')
    parser.add_argument('--sigma', type=float, default=0., help='Noise level (set > 0 for noisy data).')
    
    # Baseline
    parser.add_argument('--baseline_data', type=str, default='', help='Path to baseline results for comparison.')
    parser.add_argument('--read_baseline', action='store_true',
        help='Load baseline data from MATLAB if set; otherwise generate new data.')
    
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true', help='Enable logging to Weights & Biases.')
    parser.add_argument('--wandb_proj_name', type=str, default='BS_G_inv_multi_gpu', help='WandB project name.')
    parser.add_argument('--wandb_run_id', type=str, default="", help='WandB run ID to resume. Leave empty to start a new run.')
    
    # Loss
    parser.add_argument('--loss_criterion', type=str, default="bs_mse",
        help='Loss criterion for matched mse when K > 1: bs_mse or mse.')
    parser.add_argument('--clip_grad_norm', type=float, default=0.,
        help='Clip gradient norm if > 0.')
    
    # Run mode
    parser.add_argument('--run_mode', type=str, default="resume",
        help='Run mode: override, resume, or from_pretrained.')
    
    # Model options
    parser.add_argument('--disable_transformers', action='store_true', help='Disable transformer layers in the model. '
        'Default: Transformers are enabled.')
    
    # Swin Transformer settings
    parser.add_argument('--window_size', type=int, default=8, help='Swin Transformer window size.')
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 6], help='Transformer depths per stage.')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[2, 2], help='Number of attention heads per stage.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension.')
    
    # Output and logging
    parser.add_argument('--run_output_suffix', type=str, default='', help='Suffix for output folder name.')
    parser.add_argument('--save_every', type=int, default=5, help='Checkpoint saving interval (in epochs).')
    parser.add_argument('--print_every', type=int, default=100, help='Losses printing interval (in epochs).')
    parser.add_argument('--early_stopping', action='store_true',
        help='Enable early stopping. Set stopping parameters in the configuration.')
    
    # Performance improvements
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training.')
    parser.add_argument('--clear_gpu_cache', action='store_true', help='Clear gpu cach before training training.')
    args = parser.parse_args()

    main(args)
