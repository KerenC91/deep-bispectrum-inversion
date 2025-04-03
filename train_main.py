import torch.optim as optim
import time 
import os
import wandb
import torch 
from models.DBIModel import DBIModel
import numpy as np
from trainer import Trainer
import sys
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import create_dataset, prepare_data_loader
from torch.cuda import device_count
from utils.utils import BispectrumCalculator


def init_model(device, args, params):
    """Initialize model"""

    model = DBIModel(
         input_len = args.L,
         signals_count = args.K,
         pre_residuals=params.pre_residuals,
         pre_conv_channels=params.pre_conv_channels,
         post_residuals=params.post_residuals,
         reduce_height=params.reduce_height,
         embed_dim=args.embed_dim,
         activation=params.activation,
         window_size = args.window_size,
         patch_size = params.patch_size,
         depths = args.depths,
         num_heads = args.num_heads,
         qkv_bias = params.qkv_bias,
         qk_scale = params.qk_scale,
         drop = params.drop,
         attn_drop = params.attn_drop,
         drop_path_rate = params.drop_path_rate,
         norm_layer = params.norm_layer,
         downsample = params.downsample,
         resi_connection = params.resi_connection,
         use_transformers=not args.disable_transformers
         ).to(device)
    
    return model
    

def load_checkpoint(model, optimizer, scheduler, ckp_path, device, args, params, len_train_loader, is_distributed=False):
    """Loads checkpoint efficiently for both single-GPU and DDP with synchronized error handling."""
    map_location = "cpu" if is_distributed else f"cuda:{device}"
    epoch = 0
    error_flag = torch.tensor(0, dtype=torch.int, device=device)  # Error flag

    if device == 0 :  # Load only on Rank 0 in DDP or for Single GPU
        if os.path.exists(ckp_path):
            print('Checkpoint found')
            if args.run_mode == "override":
                print('Overriding existing checkpoint')
            elif args.run_mode == "resume" or args.run_mode == "from_pretrained":
                try:
                    checkpoint, model = load_model_safely(model, ckp_path, map_location)
                    
                    # Handle resume mode
                    if args.run_mode == "resume":
                        epoch = checkpoint['epoch']
                        if epoch >= args.epochs:  # Synchronize exit condition for all ranks
                            print(f'Error! epoch={epoch} must be smaller than args.epochs={args.epochs}')
                            error_flag += 1  # Set error flag  
                        print(f'Resuming existing run, loading checkpoint at epoch {epoch}')
                        # Load optimizer
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Load scheduler
                        if scheduler is not None:
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # Handle from_pretrained mode
                    else: #args.run_mode == "from_pretrained"
                        if scheduler is not None:
                                print(f'Starting a new run from pretrained model')
                                scheduler = set_scheduler(args.scheduler, params,
                                                          optimizer, 
                                                          args.epochs - epoch, 
                                                          args.lr, 
                                                          len_train_loader)
                    print(f"Device {device}: Checkpoint loaded.")

                except Exception as e:
                    print(f"Rank {device} encountered an error while loading checkpoint: {e}")
                    error_flag += 1  # Set error flag if loading fails

    if is_distributed:
        # Synchronize error flag across all ranks
        dist.broadcast(error_flag, src=0)

        # If error_flag is raised on rank 0, exit all ranks
        if error_flag.item() > 0:
            print(f"Rank {device} exiting due to checkpoint loading failure.")
            dist.barrier()  # Ensure all ranks synchronize before exiting
            sys.exit(1)

        # Synchronize all devices before proceeding
        dist.barrier()

        # Wrap model with DDP
        model = DDP(model, device_ids=[device], output_device=device)

        # Synchronize optimizer & scheduler states
        for param in optimizer.state.values():
            if isinstance(param, torch.Tensor):
                dist.broadcast(param, src=0)
        if scheduler is not None:
            for key, value in scheduler.state_dict().items():
                if isinstance(value, torch.Tensor):
                    dist.broadcast(value, src=0)

    return model, optimizer, scheduler, epoch

def load_model_safely(model, checkpoint_path, map_location):

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    try:
        # Try loading with strict=True (default behavior)
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("Warning: Model loading failed due to unexpected/missing keys.")
        print("Retrying with strict=False...")
        
        # Retry with strict=False to ignore mismatched keys           
        missing_keys, unexpected_keys = \
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully with strict=False.")  
        print(f"Loaded with missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    return checkpoint, model    
    


def set_optimizer(args, params, model):
       
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=params.opt_sgd_momentum,
                                    weight_decay=params.opt_sgd_weight_decay,
                                    nesterov=params.opt_sgd_nesterov)
    elif args.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=args.lr, 
                                        alpha=params.opt_rms_prop_alpha,
                                        eps=params.opt_eps)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=params.opt_adam_w_betas,
                                      eps=params.opt_adam_w_eps,
                                      weight_decay=params.opt_adam_w_weight_decay)
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      betas=params.opt_adam_betas,
                                      eps=params.opt_adam_eps,
                                      weight_decay=params.opt_adam_weight_decay)
        
    return optimizer


def set_scheduler(scheduler_name, params, optimizer, epochs, lr, len_trainloader):
    scheduler = None
    if scheduler_name != 'None':
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=params.reduce_lr_factor,
                threshold=params.reduce_lr_threshold,
                patience=params.reduce_lr_patience,
                cooldown=params.reduce_lr_cooldown)
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=params.step_lr_step_size,
                gamma=params.step_lr_gamma)
        elif scheduler_name == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=lr,
                div_factor=params.cyc_lr_div_factor,
                steps_per_epoch=len_trainloader,
                epochs=epochs,
                pct_start=params.cyc_lr_pct_start,
                anneal_strategy=params.cyc_lr_anneal_strategy)
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs * len_trainloader * params.cos_ann_lr_T_max_f) 
        elif scheduler_name == 'CyclicLR':        
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                mode=params.cyclic_lr_mode,
                base_lr=lr * params.cyclic_lr_mult_factor, 
                max_lr=lr,
                step_size_up=int(epochs * len_trainloader / 2 * params.cyclic_lr_step_size_up_mult_f),
                gamma=params.cyclic_lr_gamma) 

    return scheduler
    

def create_test_name(args):
    test_str = f'K{args.K}_N{args.L}_win{args.window_size}_bs{args.batch_size}_ep{args.epochs}_'\
                    f'tr{args.train_data_size}_val{args.val_data_size}_'\
                    f'lr_{args.lr:.1e}_{args.optimizer}_'
    if args.scheduler != 'None':
        test_str += f'{args.scheduler}'
    
    # Append user defined test name
    test_str += f'_{args.run_output_suffix}'
    
    return test_str

def train(args, params):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_implementation(0, args, params)

def train_distributed(args, params):
    # Apply ddp setup
    ddp_setup()
    
    device = int(os.environ["LOCAL_RANK"])
    print(f'Using GPU {device}')    

    train_implementation(device, args, params, is_distributed=True)


def init_folders(args, test_name):
    # Set folder to write test data to
    if not os.path.exists('output'):
            os.mkdir('output')
            
    folder_write = os.path.join('output', test_name)
    # The folder does not exist
    if not os.path.exists(folder_write):
            os.mkdir(folder_write)

    if args.read_baseline:
        # Set folder to read baseline data from
        folder_read = os.path.join('data', args.baseline_data)
        if not os.path.exists(folder_read):
            raise ValueError(f'Error! {folder_read} does not exist\n'
                  f'path={folder_read}')    
    else:
        folder_read = ''

    return folder_read, folder_write
    
    
def ddp_setup():
    device = int(os.environ["LOCAL_RANK"])
    # device = torch.device('cuda', device)
    torch.cuda.set_device(device)
    init_process_group(backend="nccl", init_method="env://")

def init_wandb(args, params, test_name, folder_write):
    wandb.login()
    if args.wandb_run_id == '':
        config = {**vars(args), **vars(params)}
        run = wandb.init(project=args.wandb_proj_name, name = f"{test_name}", config=config)
    else: #resume run
        run = wandb.init(project=args.wandb_proj_name, id=args.wandb_run_id, resume="must")
    wandb.log({"cmd_line": sys.argv})
    
    # Save wandb run id to the output folder
    np.savetxt(f'{folder_write}/wandb_run_id.csv', [wandb.run.id], fmt='%s')  
           
           
def train_implementation(device, args, params, is_distributed=False):
    torch.backends.cudnn.benchmark = True

    # Set test name    
    test_name = create_test_name(args)
 
    # Initialize args
    folder_read, folder_write = init_folders(args, test_name)
    
    # Initialize wandb
    if device == 0:
        run = None
        if args.wandb:
            init_wandb(args, params, test_name, folder_write) 
        if is_distributed:
            print(f'Distributed Training: running with {device_count()} GPUs')
        print(f'use_transformers={not args.disable_transformers}')

    # Initialize model and optimizer
    model = init_model(device, args, params)
    optimizer = set_optimizer(args, params, model)
    
    # Set helpers
    bs_calc = BispectrumCalculator(args.K, args.L, 'cpu')
    print('Set train data')
    
    # Set train dataset and dataloader
    train_dataset = create_dataset(args.train_data_size, args.K, args.L,
                                   False, args.data_mode,
                                   folder_read, bs_calc,
                                   args.sigma)
        
    train_loader = prepare_data_loader(train_dataset, args.batch_size, is_distributed)
    # Set validation dataset and dataloader 
    print('Set validation data')
    
    val_dataset = create_dataset(args.val_data_size, args.K, args.L,
                                 args.read_baseline, 'fixed',
                                 folder_read, bs_calc,
                                 args.sigma)
    
    val_loader = prepare_data_loader(val_dataset, args.batch_size, is_distributed)
    
    scheduler = set_scheduler(args.scheduler, params,
                              optimizer, 
                              args.epochs, 
                              args.lr, 
                              len(train_loader))
    
    # if exists, load from checkpoint
    ckp_path = os.path.join(f'{folder_write}', 'ckp.pt')
    
    model, optimizer, scheduler, epoch = load_checkpoint(model, 
                                                  optimizer, 
                                                  scheduler, 
                                                  ckp_path, 
                                                  device, 
                                                  args,
                                                  params,
                                                  len(train_loader),
                                                  is_distributed)
    # Initialize trainer
    trainer = Trainer(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader,
                      device=device,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      scheduler_name=args.scheduler,
                      folder_write=folder_write,
                      start_epoch=epoch,
                      args=args,
                      is_distributed=is_distributed)
    if device == 0:
        start_time = time.time()  
        print("Starting run...")
    
    # Free GPU Memory Before Training
    if args.clear_gpu_cache:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Train and evaluate
    trainer.run()
    
    if device == 0:
        end_time = time.time()
  
        print(f"Time taken to train in {os.path.basename(__file__)}:", 
              end_time - start_time, "seconds")
