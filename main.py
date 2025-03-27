#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:49:44 2024

@author: kerencohen2
"""
import torch
import argparse
from torch.cuda import device_count
import torch.multiprocessing as mp
from config.params import params
from train_main import train, train_distributed
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def is_torchrun():
    if 'LOCAL_RANK' in os.environ:
        return True
    return False

def main(args):
    replica_count = device_count() if is_torchrun() else 1
    
    if args.N % args.window_size != 0:
        raise ValueError(f'Signal size {args.N} is not evenly divisble by window size {args.window_size}. ' 
                         f'Please choose a suitable window size.')
    if replica_count > 1:
        if args.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        args.batch_size = args.batch_size // replica_count
        args.batch_size = args.batch_size // replica_count
        train_distributed(args, params)
    else:
        if torch.cuda.is_available():
            print("Running with a single GPU")
        else:
            print("GPU is not available, running with CPU")
        train(args, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverting the Bispectrum.')
    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of a signal in the dataset')
    parser.add_argument('--K', type=int, default=1, metavar='N',
            help='Number of signals to reconstruct from')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
            help='batch size')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
            help='number of epochs to run')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='f',
            help='learning rate (initial for dynamic lr, otherwise fixed)')  
    parser.add_argument('--optimizer', type=str, default="AdamW",  
                        help='The options are \"Adam\"\, \"SGD\"\, \"RMSprop\"\, \"AdamW\"\n'
                        'Please update relevant parameters in parameters file.') 
    parser.add_argument('--scheduler', type=str, default='None',
            help='\'StepLR\', \'ReduceLROnPlateau\', \'OneCycleLR\','
            ' \'CosineAnnealingLR\', \'CyclicLR\', \'Manual\'. '
            'Update configurtion parametes accordingly. '
            'default: \'None\' - no change in lr') 

    # data
    parser.add_argument('--data_mode', type=str,  default='random',
            help= 'data_mode in {\'random\'\,\'fixed\'}') 
    parser.add_argument('--train_data_size', type=int, default=5000, metavar='N',
            help='the size of the train data') 
    parser.add_argument('--val_data_size', type=int, default=100, metavar='N',
            help='the size of the validate data')  
    parser.add_argument('--sigma', type=float, default=0., metavar='f',
            help='Amount of noise in case of working with noisy dataset. Default: 0.') 
    # baseline data
    parser.add_argument('--baseline_data', type=str, default='',
            help='baseline data folder results for comparison') 
    parser.add_argument('--read_baseline', action='store_true',  
                        help='read baseline data from matlab to training set if on, '
                        'else, generate new data.')

    # wandb
    parser.add_argument('--wandb', action='store_true', help='Log data using wandb') 
    parser.add_argument('--wandb_proj_name', type=str, default='BS_G_inv_multi_gpu', 
                        help='wandb project name')
    parser.add_argument('--wandb_run_id', type=str, default="",
                        help='run id to resume running. If not provided - new run.') 
    # loss
    parser.add_argument('--loss_criterion', type=str, default="bs_mse", 
                        help='one out of \"bs_mse\", \"mse\".') 
    
    parser.add_argument('--clip_grad_norm', type=float, default=0.,  
                        help='If greater than 0: clip gradients norm with the clip_grad_norm value.') 
    parser.add_argument('--run_mode', type=str, default="resume", 
                        help='one out of \"override\", \"resume\", \"from_pretrained\" running modes.'
                        'eventhough a checkpoint exists') 

    # model 
    parser.add_argument('--disable_transformers', action='store_true', 
                        help='Disable transformers in the model. Default: Transformers are enabled.')
 
    # swin transformer
    parser.add_argument('--window_size', type=int, default=8, 
                        help='window_size')       
    parser.add_argument('--depths', type=int, nargs='+', 
                        default=[6, 6], 
                        help='depths')    
    parser.add_argument('--num_heads', type=int, nargs='+', 
                        default=[2, 2], 
                        help='num_heads')      

    #
    parser.add_argument('--run_output_suffix', type=str, default='test',
            help='folder test name to save results into') 
    parser.add_argument('--save_every', type=int, default=5, 
            help='save checkpoint every <save_every> epoch')
    parser.add_argument('--print_every', type=int, default=100,
            help='print losses every <print_every> epoch')
    parser.add_argument('--early_stopping', action='store_true', 
                        help='early stopping after early_stopping times. '
                        'Update early_stopping in configuration') 

    # mixed precision
    parser.add_argument('--fp16', action='store_true', 
                        help='Use mixed percision if on.')
    
    args = parser.parse_args()

    main(args)
