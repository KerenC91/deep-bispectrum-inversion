#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:36:09 2025

@author: kerencohen2
"""

class DictParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
inference_args = DictParams(
    L = 24,
    K = 2,
    window_size = 6, 
    depths = [6, 6],                      
    num_heads = [2, 2], 
    embed_dim = 256,
    data_mode = 'random', # 'random' / 'fixed'
    data_size=100,
    disable_transformers = False,
    from_pretrained = False,
    activation = 'LeakyReLU',
    sigma=0.,
    )

inference_params = DictParams(

    
    qkv_bias = False,
    qk_scale = False,
    drop = 0.,
    attn_drop = 0.,
    drop_path_rate = 0.1,
    norm_layer = True,
    downsample = False,
    resi_connection = '1conv',
    patch_size=1, #'patch size used in training SwinIR. '
                  #'Just used to differentiate two different settings in Table 2 of the paper. '
                  #'Images are NOT tested patch by patch.'
    pre_conv_channels = [8, 32], 
    reduce_height = [4, 3, 3],      
    pre_residuals = 11,
    post_residuals = 14,   
    activation = 'LeakyReLU',      
    )


