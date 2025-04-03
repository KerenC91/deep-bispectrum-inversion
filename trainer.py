import os
import wandb
import torch
from utils.utils import BispectrumCalculator, compute_cost_matrix, greedy_match
from config.params import params
import numpy as np


class Trainer:
    def __init__(self, model,
                 train_loader,
                 val_loader,
                 device,
                 optimizer,
                 scheduler,
                 scheduler_name,
                 folder_write,
                 start_epoch,
                 args,
                 is_distributed=False):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.target_len = args.L
        self.signals_count = args.K
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.model = model
        self.is_distributed = is_distributed
        self.wandb_flag = args.wandb
        self.start_epoch = start_epoch
        self.epoch = 0
        self.prev_val_loss = torch.inf
        self.early_stopping = args.early_stopping
        self.es_cnt = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.data_mode = args.data_mode
        self.bs_calc = BispectrumCalculator(self.signals_count, self.target_len, self.device).to(self.device)
        self.folder_write = folder_write
        self.clip = args.clip_grad_norm
        self.loss_criterion = args.loss_criterion
        self.is_master = (device == 0)
        self.min_ckp_val_loss = torch.inf
        self.min_loss_epoch = 0
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.fp16)
        self.scaler = torch.amp.GradScaler(device_type, enabled=args.fp16)
        self.sigma = args.sigma
        self.fp16 = args.fp16

    def _loss(self, pred, target):
        total_loss = 0.

        # Compute avergae bispectrum mse loss
        bs_pred, _ = self.bs_calc(pred)
        bs_target, _ = self.bs_calc(target)
        bs_mse_loss = torch.norm(bs_pred - bs_target) ** 2 / torch.norm(bs_target) ** 2

        # Compute matched mse loss according to loss criterion
        if self.signals_count > 1:

            matched_loss = self._compute_matched_loss(pred, target)

            total_loss = (1 - params.loss_alpha) * bs_mse_loss + params.loss_alpha * matched_loss
        else:
            total_loss = bs_mse_loss

        return total_loss

    def _compute_matched_loss(self, pred, target):
        """
        Computes MSE only for the K matched pairs.
        """
        B, K, _ = pred.shape
        loss = 0

        cost_matrix = compute_cost_matrix(pred, target, self.bs_calc, self.loss_criterion, self.fp16)
        matches = greedy_match(cost_matrix)  # Get matched pairs

        for b in range(B):
            for i, j in matches[b]:
                loss += cost_matrix[b, i, j]

        return loss / (B * K)  # Normalize over BxK pairs

    def _run_batch(self, source, target, data_mode='fixed'):

        if data_mode == 'random':
            target, source = self.generate_random_data()

        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)

        with self.autocast:  # Enable Mixed Precision
            # Forward pass
            output = self.model(source)  # reconstructed signal
            # Loss calculation  
            loss = self._loss(output, target)

        return loss

    def generate_random_data(self):
        target = torch.randn(self.batch_size, self.signals_count, self.target_len)

        if self.sigma:
            data = target + self.sigma * torch.randn(self.batch_size, self.signals_count, self.target_len)
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
        self.update_scheduler_after_epoch(avg_loss)

        return avg_loss

    def update_scheduler_after_batch(self):
        if self.scheduler_name in ['OneCycleLR', 'CosineAnnealingLR', 'CyclicLR']:
            self.scheduler.step()

    def update_scheduler_after_epoch(self, loss):
        if self.scheduler_name == 'Manual':
            if self.epoch in params.manual_epochs_lr_change:
                self.optimizer.param_groups[0]['lr'] *= params.manual_lr_f
        elif self.scheduler_name == 'StepLR':
            self.scheduler.step()
        elif self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(loss)

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

    def log_wandb(self, train_loss, val_loss):
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        wandb.log({"lr": self.optimizer.param_groups[0]['lr']})

    def check_early_stopping(self, val_loss):
        if self.prev_val_loss < val_loss:
            self.es_cnt += 1
        else:
            self.es_cnt = 0  # Reset counter if performance improves

        if self.es_cnt >= params.early_stopping_count:
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
                # save checkpoint
                if self.epoch == 1 or self.epoch % self.save_every == 0:
                    if val_loss < self.min_ckp_val_loss:
                        # Update the new minimum
                        self.min_ckp_val_loss = val_loss
                        self.min_loss_epoch = self.epoch
                        # Save new checkpoint
                        self._save_checkpoint()

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
                        self.log_wandb(train_loss, val_loss)

                # early stopping - stop early if early_stopping is on
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
