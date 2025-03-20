import numpy as np


class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hparams = HParams(
    
    #####################################
    # data config 
    #####################################
    sigma = 0.1,
    ##########################
    # scheduler config
    ##########################
    # Manual:
    manual_lr_f = 0.1,    
    manual_epochs_lr_change = [2000, 3000, 4000, 5000, 6000],
        
    # ReduceLROnPlateau 
    reduce_lr_mode='min',
    reduce_lr_factor = 0.5,
    reduce_lr_threshold = 1e-4,
    reduce_lr_patience = 5,
    reduce_lr_cooldown = 0,

    # StepLR - every step_size epochs decrease by lr gamma factor
    step_lr_step_size = 2000,#1000, 
    step_lr_gamma = 0.94,
    
    # OneCycleLR - perform one cycle of learning. 
    # epochs and steps per epochs are defined in the code
    # cyc_lr_max_lr = 1e-2,
    cyc_lr_pct_start = 0.562,
    cyc_lr_anneal_strategy = 'cos',
    cyc_lr_three_pahse= True,
    cyc_lr_div_factor = 16,#25
	#"cyc_lr_epochs": num_epochs,
	#"cyc_lr_steps_per_epoch": len(train_loader),
    
    # CosineAnnealingLR - used as:
    # cos_ann_lr_T_max = int(num_epochs * len(train_loader) * cos_ann_lr_T_max_f)
    # performs (cos_ann_lr_T_max_f / 2) cosine periods
    cos_ann_lr_T_max_f = 0.1,
    
    # CyclicLR
    # cyclic_lr_step_size_up = int(num_epochs * len(train_loader) / 2 / cyclic_lr_step_size_up_f)
    # Performs cyclic_lr_step_size_up_f traingle periods
    cyclic_lr_base_lr=1e-4, 
    # cyclic_lr_max_lr=1e-2,
    cyclic_lr_mode="triangular",
    cyclic_lr_step_size_up_f=3,
    cyclic_lr_gamma=1,
    
    ##########################
    # optimizer
    ##########################
    # RMSProp
    opt_rms_prop_alpha = 0.99,
    # SGD
    opt_sgd_momentum = 0.9,
    opt_sgd_weight_decay = 1e-4,
    # AdamW
    opt_adam_w_betas=(0.9, 0.999),
    opt_adam_w_weight_decay=1e-2,
    opt_adam_w_eps = 1e-8,
    # Adam
    opt_adam_betas=(0.9, 0.999),
    opt_adam_eps = 1e-8,
    opt_adam_weight_decay=0.0,
    
    # all optimizer params
    opt_eps = 9.606529741408894e-07,

    ##########################
    # CNN params
    ##########################
    last_ch = 256, # last ch of pre conv. for all models: 8, for model3: 256

    pre_conv_channels = [8, 32],#[8, 32, 256], 
                        #layer_channels list of values on each of heads
    reduce_height = [4, 3, 3], # RELEVANT FOR MODEL2, 3 ONLY
                    #relevant only for model2 - [count kernel stride]
                    #for reducing height in tensor: BXCXHXW to BXCX1XW
    pre_residuals = 9,#11, 
    up_residuals = 8,#3,    
    post_residuals = 2,#14,
    activation = 'LeakyReLU',
    ##########################
    # additional params
    ##########################
    early_stopping_count = 100,
    # comparison with baseline
    data_root = '../data',
)

