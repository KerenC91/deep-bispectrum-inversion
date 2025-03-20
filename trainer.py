import os
import wandb
import torch 
from utils.utils import BispectrumCalculator, BatchAligneToReference, align_to_reference
from config.hparams import hparams
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, all_reduce
import gc
import sys
import pdb

class Trainer:
    def __init__(self, model, 
                        train_loader, 
                        val_loader, 
                        train_dataset,
                        val_dataset,
                        wandb_flag,
                        device,
                        optimizer,
                        optimizer_name,
                        scheduler,
                        scheduler_name,
                        folder_write,
                        start_epoch,
                        args,
                        is_distributed=False):
        self.device = device 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.target_len = args.N
        self.signals_count = args.K
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.model = model
        self.is_distributed = is_distributed
        self.wandb_flag = wandb_flag
        self.start_epoch = start_epoch
        self.epoch = 0
        self.prev_val_loss = torch.inf
        self.early_stopping = args.early_stopping
        self.es_cnt = 0
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.read_baseline = args.read_baseline
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.data_mode = args.data_mode
        self.loss_f = self._loss
        self.bs_calc = BispectrumCalculator(self.signals_count, self.target_len, self.device).to(self.device)
        self.aligner = BatchAligneToReference(self.device).to(self.device)
        self.folder_write = folder_write
        self.loss_method = args.loss_method
        self.clip = args.clip_grad_norm
        self.loss_criterion = args.loss_criterion
        self.is_master = (device == 0)
        self.log_level = args.log_level
        self.min_ckp_val_loss = torch.inf
        self.min_loss_epoch = 0        
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.fp16)
        self.scaler = torch.amp.GradScaler(device_type, enabled=args.fp16)
        self.noisy = args.noisy
        
    def _loss(self, pred, target):
        total_loss = 0.
        
        if self.loss_criterion == "sc":
            bs_pred, _ = self.bs_calc(pred)
            bs_target, _ = self.bs_calc(target)
            loss_sc = self._loss_sc(bs_pred, bs_target)
            total_loss = loss_sc
        # elif self.loss_criterion == "l1":
        #     loss_l1_aligned = self._loss_l1(pred, target)
        #     total_loss = loss_l1_aligned  
        if self.loss_criterion == "mse":
            loss_mse_aligned = self._loss_MSE(pred, target)
            total_loss = loss_mse_aligned

        # if self.signals_count > 1:
        #     norm_loss = self._compute_matched_loss(pred, target)
        
        # total_loss += norm_loss
        
        return total_loss
    
    def _switch_position(self, pred, target):
        # pdb.set_trace()
        switch = False
        if self.loss_criterion == "sc":
            bs_pred, pred = self.bs_calc(pred, "sum")
            bs_target, target = self.bs_calc(target, "sum")
            _, switch = self._switch_criterion(bs_pred, bs_target)
        elif self.loss_criterion == "l1":
            _, switch = self._switch_criterion_l1_aligned(pred, target)
        elif self.loss_criterion == "mse":
            _, switch = self._switch_criterion_mse_aligned(pred, target)
        if switch:
            pred = torch.flip(pred, dims=(-2,))
        
        return pred
    
   
    def _loss_sc(self, bs_pred, bs_gt, method="average"):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
            || BS(rec_s) - BS(s) ||_F / || BS(s) ||_F.

        """
        if method == "sum":
            sh = bs_pred.shape
            loss = torch.mean(
                        torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2/ \
                            torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2))**2)
        else:
            loss = torch.norm(bs_pred - bs_gt)**2 / torch.norm(bs_gt)**2
            
        return loss
    
    def _compute_matched_loss(self, pred, target):
        """
        Computes MSE only for the K matched pairs.
        """
        selected_indices = self._optimal_assignment(pred, target)  # Get matched pairs
        total_loss = 0
        
        # i, j go from 0 to K-1. We have K pairs
        for (b, i, j) in selected_indices:
            aligned_pred, _ = align_to_reference(pred[b, j], target[b, i])  # Keep gradients
            mse_loss = torch.nn.functional.mse_loss(aligned_pred, target[b, i])  # Differentiable loss
            total_loss += mse_loss  # Only differentiating the selected matches
    
        return total_loss / len(selected_indices)  # Normalize over BxK pairs

    

    def _compute_cost_matrix(self, pred, target):
        """
        Compute the KxK cost matrix where each entry (i, j) is the minimal L2 distance
        between x_i and the best circularly shifted x_pred_j.
        """
        B, K, N = pred.shape
        cost_matrix = torch.zeros((B, K, K))
        
        for b in range(B):
            for i in range(K):
                for j in range(K):
                    pred[b, j], _ = align_to_reference(pred[b,j], target[b,i])
                    cost_matrix[b, i, j] = torch.norm(pred[b,j] - target[b,i])  # Compute L2 distance
    
        return cost_matrix

    def _optimal_assignment(self, pred, target):
        """
        Greedy method: Iteratively pick the smallest cost, remove row and column.
        Differentiable version: Returns indices instead of summing loss.
        
        Returns:
            selected_indices: List of (row, col) index pairs.
        """
        B, K, N = pred.shape
        cost_matrix = self._compute_cost_matrix(pred, target)  # (B, K, K)
        

        selected_indices = []
    
        for b in range(B):
            selected_rows = set()
            selected_cols = set()
            for _ in range(K):
                min_value = float("inf")
                min_row, min_col = -1, -1
        
                for i in range(K):
                    if i in selected_rows:
                        continue
                    for j in range(K):
                        if j in selected_cols:
                            continue
                        if cost_matrix[b, i, j] < min_value:
                            min_value = cost_matrix[b, i, j]
                            min_row, min_col = i, j
        
                # Store selected pairs
                selected_indices.append((b, min_row, min_col))
                selected_rows.add(min_row)
                selected_cols.add(min_col)
    
        return selected_indices  # Return selected matches only
    

    def _switch_criterion(self, bs_pred, bs_gt):
        # for sum method only
        sh = bs_pred.shape
        reversed_bs_pred = torch.flip(bs_pred, dims=(1,))
        loss1 = torch.mean(
                    torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2 / \
                        torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))**2
        loss2 = torch.mean(
                    torch.norm((reversed_bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2 / \
                        torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))**2
        # get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch
    
    def _switch_criterion_l1_aligned(self, pred, target):
        # pdb.set_trace()
        # for sum method only
        # Get shape
        sh = pred.shape
        # Get prediction after flipping locations
        reversed_pred = torch.flip(pred, dims=(1,))
        
        # Align pred to target per signal 1,...,K (original order 0,1)
        pred, _ = self.aligner(pred, target)
        loss1 = self._loss_l1(pred, target)
        
        # Align reversed_pred to target per signal 1,...,K (flipped order 1,0)
        reversed_pred, _ = self.aligner(reversed_pred, target)
        loss2 = self._loss_l1(reversed_pred, target)
        
        # Get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # Get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch

    def _switch_criterion_mse_aligned(self, pred, target):
        # for sum method only
        sh = pred.shape
        reversed_pred = torch.flip(pred, dims=(1,))
        
        pred, _ = self.aligner(pred, target)
        loss1 = self._loss_MSE(pred, target)
        
        reversed_pred, _ = self.aligner(reversed_pred, target)
        loss2 = self._loss_MSE(reversed_pred, target)
        # get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)

    def _loss_rel_MSE(self, pred, target):

        return torch.mean(
                    torch.norm(pred - target, dim=(0, 2))**2 / \
                    torch.norm(target, dim=(0, 2))**2)

    def _loss_l1(self, pred, target):
 
        criterion = torch.nn.L1Loss()          

        return criterion(pred, target)
    
        # target - ground truth image, source - Bispectrum of ground truth image
        # might be multiple targets and sources (batch size > 1)
        
    def _loss_MSE(self, pred, target):

        criterion = torch.nn.MSELoss()  
        
        return criterion(pred, target)
        
    def _run_batch(self, source, target, data_mode='fixed'):
        
        if data_mode == 'random':
            target, source = self.generate_random_data()

        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)

        with self.autocast:  # Enable Mixed Precision
            # Forward pass
            output = self.model(source) # reconstructed signal

            if self.signals_count > 1:
                output = self._switch_position(output, target)
                 
            # Loss calculation  
            loss = self.loss_f(output, target)

        return loss
            
    def generate_random_data(self):
        target = torch.randn(self.batch_size, self.signals_count, self.target_len)
    
        if self.noisy:
            data = target + hparams.sigma * torch.randn(self.batch_size, self.signals_count, self.target_len)
            source, data = self.bs_calc(data)
        else:
            source, target = self.bs_calc(target)
        
        return target, source
    
    def _save_checkpoint(self):
        if not os.path.exists(self.folder_write):
            os.makedirs(self.folder_write)
        torch.save({'epoch': self.epoch,
            'model_state_dict': 
                self.model.module.state_dict() if self.is_distributed 
                else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if self.scheduler is None
                else self.scheduler.state_dict()}, 
            f'{self.folder_write}/ckp.pt')
 
        if self.wandb_flag:
            wandb.save(f'{self.folder_write}/ckp.pt', base_path=f'{self.folder_write}')
          
        
    def _run_epoch_train(self):
        total_loss = 0.
        for idx, (sources, targets) in self.train_loader:
            with torch.autograd.set_detect_anomaly(True):

                # zero grads
                self.optimizer.zero_grad()
                # forward pass + loss computation
                loss = self._run_batch(sources, targets, self.data_mode)
    
                # backward passs
                self.scaler.scale(loss).backward()
                # clip gradients
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                # optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # update avg loss 
                total_loss += loss.item()
                # scheduler step after batch
                self.update_scheduler_after_batch()
            
        avg_loss = total_loss / len(self.train_loader)
        
        # scheduler step after epoch
        self.update_scheduler_after_epoch()
                
        return avg_loss

    def update_scheduler_after_batch(self):
        if self.scheduler_name in ['OneCycleLR', 'CosineAnnealingLR', 'CyclicLR']:
            self.scheduler.step()
                
    def update_scheduler_after_epoch(self):
        if self.scheduler_name == 'Manual':
            if self.epoch in hparams.manual_epochs_lr_change:
                self.optimizer.param_groups[0]['lr'] *= hparams.manual_lr_f 
        elif self.scheduler_name == 'StepLR':
            self.scheduler.step()
        elif self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(avg_loss)
     
    def _run_epoch_validate(self):
        total_loss = 0.
        
        for idx, (sources, targets) in self.val_loader:
            with torch.no_grad():
                # forward pass + loss computation
                loss = self._run_batch(sources, targets)

                # update avg loss 
                total_loss += loss.item()
            
        avg_loss = total_loss / len(self.val_loader)
            
        return avg_loss
  
    def log_wandb(self):
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        wandb.log({"lr": self.optimizer.param_groups[0]['lr']})
    
    def check_early_stopping(self, val_loss):
        if self.prev_val_loss < val_loss:
            self.es_cnt += 1
        else:
            self.es_cnt = 0  # Reset counter if performance improves
        
        if self.es_cnt >= hparams.early_stopping_count:
            print(f'Early stopping at epoch {self.epoch}, after {self.es_cnt} times\n'
                  f'prev_val_loss={self.prev_val_loss}, curr_loss={val_loss}')
            self._save_checkpoint()
            
            return True  # Signal to stop
        
        return False

    # one epoch of training           
    def train(self):
        # Set the model to training mode
        self.model.train()
        
        avg_loss = self._run_epoch_train()
            
        return avg_loss
    
    # one epoch of validation           
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()

        avg_loss = self._run_epoch_validate()
            
        return avg_loss

    # one epoch of testing 
    def test(self):
        return 0
    
        
    def run(self):
       
        should_stop = torch.tensor([0], device=self.device)  # 0 = continue, 1 = stop

        for self.epoch in range(self.start_epoch + 1, self.epochs + 1):
            # train             
            train_loss = self.train()
            # validate
            val_loss = self.validate()

            
            if np.isnan(train_loss).any() or np.isnan(val_loss).any():
              raise RuntimeError(f'Detected NaN loss at epoch {self.epoch}.')
              
            if self.is_master:
                    
                # print losses
                if self.epoch == 1 or self.epoch % self.print_every == 0:
                    print(f'-------Epoch {self.epoch}/{self.epochs}-------')
                    print(f'Total Train loss: {train_loss:.6f}')
                    print(f'Total Validation loss: {val_loss:.6f}')
                    print(f'The minimal validation loss is {self.min_ckp_val_loss} from epoch {self.min_loss_epoch}.')   

                    # print lr when using scheduler
                    if self.scheduler_name != 'None':
                        last_lr = self.optimizer.param_groups[0]['lr']
                        print(f'lr: {last_lr}')
                    
                    # log losses with wandb
                    if self.wandb_flag:
                        self.log_wandb()
                        
                # save checkpoint
                if self.epoch == 1 or self.epoch % self.save_every == 0:
                    if val_loss < self.min_ckp_val_loss: 
                        # Update the new minimum
                        self.min_ckp_val_loss = val_loss
                        self.min_loss_epoch = self.epoch
                        # Save new checkpoint
                        self._save_checkpoint()
                      
                #early stopping - stop early if early_stopping is on
                if self.early_stopping:
                    stop = self.check_early_stopping(val_loss)
                    should_stop[0] = int(stop)

                self.prev_val_loss = val_loss
            
            # Broadcast stopping signal to all ranks
            if self.is_distributed:
                torch.distributed.broadcast(should_stop, src=0)
            
            # All processes check whether to stop
            if should_stop.item() == 1:
                if self.is_master:
                    print(f"[Epoch {self.epoch}] Early stopping triggered.")
                break  # Exit training loop
        
        # test
        with torch.no_grad():
            test_loss = self.test()
            if self.is_master:
                print(f'Test loss l1: {test_loss:.6f}')
