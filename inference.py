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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import sys

sys.path.append(parent_dir)
sys.path.append(f'{parent_dir}/config')

import torch
import matplotlib.pyplot as plt
from utils.utils import align_to_reference, BispectrumCalculator, compute_cost_matrix, greedy_match
from train_main import init_model
from dataset import read_dataset_from_baseline, create_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config.inference_params import inference_params, inference_args
import argparse
import torch.nn.functional as F



torch.set_default_dtype(torch.float64)


def evaluate(device, dataloader, baseline, model, bs_calc, args, output_path):

    avg_rel_mse_err = 0.
    avg_l1_err = 0.
    avg_mse_err = 0.

    b_avg_rel_mse_err = 0.
    b_avg_l1_err = 0.
    b_avg_mse_err = 0.
    
    min_rel_mse_err = np.inf
    min_ind = 0.
    max_rel_mse_err = 0.
    max_ind = 0.

    # loop over signals
    for idx, (source, target) in dataloader:
    
        i = idx.item()
        source = source.to(device)
        target = target.to(device)
        
        # Set output folder per sample
        folder_write = os.path.join(output_path, f'sample{i + 1}')
        if not os.path.exists(folder_write):
            os.mkdir(folder_write)
      
        # Pass the sample's bispectrum through the model
        output = model(source)
        
        # Compute matched loss
        cost_matrix = compute_cost_matrix(output, target, bs_calc)
        matches = greedy_match(cost_matrix)[0]  # Get matched pairs
        
        if baseline is not None:
            bcost_matrix = compute_cost_matrix(baseline[i].unsqueeze(0), target, bs_calc)
            bmatches = greedy_match(bcost_matrix)[0]  # Get matched pairs
            matches = match_inidices_to_baseline(matches, bmatches)

        l1_err = 0.
        mse_err = 0.
        b_l1_err = 0.
        b_mse_err = 0.
        
        for k, inds in enumerate(matches, start=1):
            curr_target = target[0][inds[0]]
            output[0][inds[1]], _ = align_to_reference(output[0][inds[1]], curr_target)
            curr_output = output[0][inds[1]]
            l1_err += F.l1_loss(curr_output, curr_target, reduction='mean')
            mse_err += F.mse_loss(curr_output, curr_target, reduction='mean')

            if baseline is not None:
                # Get index fitted to target in place inds[0]
                # No need to align baseline, already aligned by the baseline code
                curr_baseline = baseline[i][inds[2]]
                b_l1_err += F.l1_loss(curr_baseline, curr_target, reduction='mean')
                b_mse_err += F.mse_loss(curr_baseline, curr_target, reduction='mean')
            else:
                aligned_baseline = None
            plot_comparison(i, k, folder_write, curr_target, curr_output, curr_baseline)
        
        rel_mse_err = torch.norm(output[0] - target[0]) ** 2 / torch.norm(target[0]) ** 2
        b_rel_mse_err = torch.norm(baseline[i] - target[0]) ** 2 / torch.norm(target[0]) ** 2
        
        print(f'sample{i + 1}: rel_mse_err={rel_mse_err:.15f}, b_rel_mse_err={b_rel_mse_err:.15f}, ')

        # Update target to output error
        rel_mse_err = rel_mse_err.item()
        l1_err = l1_err.item() / args.K
        mse_err = mse_err.item() / args.K

        # Update minimal and maximal errors
        if rel_mse_err < min_rel_mse_err:
            min_rel_mse_err = rel_mse_err
            min_ind = i
        if rel_mse_err > max_rel_mse_err:
            max_rel_mse_err = rel_mse_err
            max_ind = i

        # Write errors to file
        np.savetxt(f'{folder_write}/rel_mse_err.csv', [rel_mse_err])
        np.savetxt(f'{folder_write}/l1_err.csv', [l1_err])
        np.savetxt(f'{folder_write}/mse_err.csv', [mse_err])

        if baseline is not None:
            b_rel_mse_err = b_rel_mse_err.item()
            b_l1_err = b_l1_err.item() / args.K
            b_mse_err = b_mse_err.item() / args.K
            np.savetxt(f'{folder_write}/b_rel_mse_err.csv', [b_rel_mse_err])
            np.savetxt(f'{folder_write}/b_l1_err.csv', [b_l1_err])
            np.savetxt(f'{folder_write}/b_mse_err.csv', [b_mse_err])


        avg_rel_mse_err += rel_mse_err
        avg_l1_err += l1_err
        avg_mse_err += mse_err

        if baseline is not None:
            b_avg_rel_mse_err += b_rel_mse_err
            b_avg_l1_err += b_l1_err
            b_avg_mse_err += b_mse_err

    avg_rel_mse_err /= args.data_size
    avg_l1_err /= args.data_size
    avg_mse_err /= args.data_size

    if baseline is not None:
        b_avg_rel_mse_err /= args.data_size
        b_avg_l1_err /= args.data_size
        b_avg_mse_err /= args.data_size
        
    print(f'total avg: avg_rel_mse_err={avg_rel_mse_err:.15f}, avg_l1_err={avg_l1_err:.15f},' +
          f' avg_mse_err={avg_mse_err:.15f}')
    print(f'minimal relative mse error={min_rel_mse_err:.15f} obtained by sample{min_ind + 1}')
    print(f'maximal relative mse error={max_rel_mse_err:.15f} obtained by sample{max_ind + 1}')

    np.savetxt(f'{output_path}/avg_rel_mse_err.csv', [avg_rel_mse_err])
    np.savetxt(f'{output_path}/avg_l1_err.csv', [avg_l1_err])
    np.savetxt(f'{output_path}/avg_mse_err.csv', [avg_mse_err])

    if baseline is not None:
        print(f'baseline total avg: b_avg_rel_mse_err={b_avg_rel_mse_err:.15f}, b_avg_l1_err={b_avg_l1_err:.15f},' +
              f' b_avg_mse_err={b_avg_mse_err:.15f}')
        np.savetxt(f'{output_path}/b_avg_rel_mse_err.csv', [b_avg_rel_mse_err])
        np.savetxt(f'{output_path}/b_avg_l1_err.csv', [b_avg_l1_err])
        np.savetxt(f'{output_path}/b_avg_mse_err.csv', [b_avg_mse_err])


def match_inidices_to_baseline(ot_matches, bt_matches):
    '''
    Parameters
    ----------
    ot_matches : list of length K, containing (i, j) match indices for target and output respectively
    bt_matches : list of length K, containing (i, j) match indices for target and baseline respectively
        DESCRIPTION.
    
    Returns a unified list of matches, each containing a tuple of indices (target, pred, baseline)
    -------
    None.
    
    '''
    matches = []
    K = len(ot_matches)
    used_ks = set()
    
    for k1 in range(K): #looping over ot_matches
        for k2 in range(K): #looping over bt_matches
            if k2 in used_ks: 
                continue
            if ot_matches[k1][0] == bt_matches[k2][0]: #looking for a match in target index
                matches.append((ot_matches[k1][0], ot_matches[k1][1], bt_matches[k2][1]))
                used_ks.add(k2)
                break
    
    return matches  # list of length K of (i, j, l) tuples

        
def plot_comparison(i, k, folder_write, s_k, s_k_pred, s_baseline):
    folder_k = os.path.join(folder_write, f'{k}')
    if not os.path.exists(folder_k):
        os.mkdir(folder_k)
        
    # fig
    fig_path = os.path.join(folder_k, f'comp_s{k}_s{k}_pred.jpg')
    plt.figure(figsize=(9, 5))
    plt.title(f'Comparison between s{k}, s{k}_pred, sample{i + 1}')
    plt.plot(s_k.cpu().detach().numpy(), label=f's{k}', color='tab:blue')
    plt.plot(s_k_pred.cpu().detach().numpy(), label=f's{k}_pred', color='tab:orange', linestyle='dashed')
    if s_baseline is not None:
        plt.plot(s_baseline.cpu().detach().numpy(), label=f's{k}_baseline', color='tab:red', linestyle='dotted', linewidth=2.5)

    
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(fig_path)        
    plt.close()   


def main(args, params, model_dir, data_dir):

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Init helpers
    bs_calc = BispectrumCalculator(args.K, args.L, device).to('cpu')
    
    if os.path.exists(data_dir):
        print(f'Reading dataset from baseline folder...')
        read_baseline = True
    else:
        print(f'Warning: data_dir does not exist: {data_dir}, creating a new dataset.')
        read_baseline = False
        args.data_mode = 'fixed'
        
    # Create test dataset
    dataset = create_dataset(args.data_size,
                             args.K, 
                             args.L, 
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
    if read_baseline:
        _, baseline = read_dataset_from_baseline(data_dir,
                                                 args.data_size,
                                                 args.K,
                                                 args.L,
                                                 args.sigma,
                                                 "x_est")
        baseline = baseline.to(device)
    else:
        baseline = None

        # Set model path
    model_path = os.path.join(model_dir, 'ckp.pt')

    # Load the model   
    model = init_model(device, args, params)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)    
    
    with torch.no_grad():
        evaluate(device, dataloader, baseline, model, bs_calc, args=args, output_path=model_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs inference on a dataset')
    parser.add_argument('--model_dir', type=str,
                        help='directory containing a trained model (or full path to weights.pt file)')  # test_folder
    parser.add_argument('--data_dir', type=str, help='directory containing validation dataset',
                        default='')  # baseline_data_folder

    args = parser.parse_args()
    main(args=inference_args, params=inference_params, model_dir=args.model_dir, data_dir=args.data_dir)


