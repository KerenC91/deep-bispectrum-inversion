
class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


params = Params(
    ##########################
    # loss
    ##########################
    loss_alpha=1.,
    ##########################
    # scheduler
    ##########################
    # Manual:
    manual_lr_f=0.1,    
    manual_epochs_lr_change=[2000, 3000, 4000, 5000, 6000],
        
    # ReduceLROnPlateau 
    reduce_lr_factor=0.5,
    reduce_lr_threshold=1e-4,
    reduce_lr_patience=3,
    reduce_lr_cooldown=2,

    # StepLR - every step_size epochs decrease by lr gamma factor
    step_lr_step_size=200,
    step_lr_gamma=0.5,
    
    # OneCycleLR - perform one cycle of learning. 
    # epochs and steps per epochs are defined in the code
    cyc_lr_pct_start=0.562,
    cyc_lr_anneal_strategy='cos',
    cyc_lr_div_factor=57,
    
    # CosineAnnealingLR - used as:
    # cos_ann_lr_T_max=int(num_epochs * len(train_loader) * cos_ann_lr_T_max_f)
    cos_ann_lr_T_max_f=1.,
    
    # CyclicLR
    # cyclic_lr_step_size_up=int(num_epochs * len(train_loader) / 2 * cyclic_lr_step_size_up_mult_f)
    # Performs cyclic_lr_step_size_up_f traingle periods
    cyclic_lr_mult_factor=1e-2,
    cyclic_lr_mode="triangular",
    cyclic_lr_step_size_up_mult_f=1.,
    cyclic_lr_gamma=1,
    
    ##########################
    # optimizer
    ##########################
    # RMSProp
    opt_rms_prop_alpha=0.99,
    # SGD
    opt_sgd_momentum=0.9,
    opt_sgd_weight_decay=1e-4,
    opt_sgd_nesterov=True,
    # AdamW
    opt_adam_w_betas=(0.9, 0.999),
    opt_adam_w_weight_decay=1e-2,
    opt_adam_w_eps=1e-8,
    # Adam
    opt_adam_betas=(0.9, 0.999),
    opt_adam_eps=1e-8,
    opt_adam_weight_decay=0.0,
    
    # all optimizers' params
    opt_eps=9.606529741408894e-07,

    ##########################
    # DNN
    ##########################
    # Swin Transformers
    qkv_bias=False,
    qk_scale=False,
    drop=0.,
    attn_drop=0.,
    drop_path_rate=0.1,
    norm_layer=True,
    downsample=False,
    resi_connection='1conv',
    patch_size=1,  # 'patch size used in training SwinIR. '
                   # 'Just used to differentiate two different settings in Table 2 of the paper. '
                   # 'Images are NOT tested patch by patch.'
    # Additional parameters
    pre_conv_channels=[8, 32],
    reduce_height=[4, 3, 3],  # for reducing height in tensor: BXCXHXW to BXCX1XW
    pre_residuals=11,
    post_residuals=14,
    activation='LeakyReLU',
    ##########################
    # additional params
    ##########################
    early_stopping_count=100,
)