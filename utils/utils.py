import torch
import numpy as np
import torch
import torch.nn as nn
import math

device_type = "cuda" if torch.cuda.is_available() else "cpu"

  
def clculate_bispectrum_efficient(x, normalize=False):
    """
    

    Parameters
    ----------
    x : torch N size, float
        signal.
    Returns
    -------
    Bx : torch NXNX1 size, complex-float
        Bispectrum.

    """
    with torch.autocast(device_type=device_type, enabled=False):
        y = torch.fft.fft(x)
        circulant = lambda v: torch.cat([f := v, f[:-1]]).unfold(0, len(v), 1).flip(0)
        C = circulant(torch.roll(y, -1))
        Bx = y.unsqueeze(1) @ y.conj().unsqueeze(0)
        Bx = Bx * C
        
        if normalize:
            eps = 1e-8
            Bx_factor = torch.pow(torch.abs(Bx), 2/3) + eps
            Bx = Bx / Bx_factor
    return Bx



class BispectrumCalculator(nn.Module):
    def __init__(self, targets_count, target_len, device):
        super().__init__()
        self.calculator = clculate_bispectrum_efficient
        self.targets_count = targets_count
        self.target_len = target_len
        self.device = device
        self.channels = 2
        self.height = target_len
        self.width = target_len
        
    def _create_data(self, target):
        with torch.autocast(device_type=device_type, enabled=False):
            bs = self.calculator(target.to(dtype=torch.float32, copy=True))
            source = torch.stack([bs.real.float(), bs.imag.float()], dim=0)
               
        return source, target 
    # target: signal 1Xtarget_len
    # source: bs     2Xtarget_lenXtarget_len
    def forward(self, target, method="average"):
        batch_size = target.shape[0]
        # Iterate over the batch dimension using indexing
        # if method == "sum":#Unused
        #     source = torch.zeros(batch_size, self.targets_count, self.channels, self.height, self.width).to(self.device)
      
        #     for i in range(batch_size):
        #         for j in range(self.targets_count):
        #             source[i][j], target[i][j] = self._create_data(target[i][j])
        if method == "batched":
            source = torch.zeros(batch_size, self.channels, self.height, self.width).to(self.device)
      
            for i in range(batch_size):
                source[i], target[i] = self._create_data(target[i])
        else: #average
            source = torch.zeros(batch_size, self.channels, self.height, self.width).to(self.device)
            s = torch.zeros(self.channels, self.height, self.width).to(self.device)
            
            for i in range(batch_size):
                for j in range(self.targets_count):
                    s, target[i][j] = self._create_data(target[i][j])
                    source[i] += s.to(self.device)
                source[i] /= self.targets_count            
            #add for sum loss metric
        return source, target  # Stack processed vectors



class BatchAligneToReference(nn.Module):
    def __init__(self, device):
        super().__init__()
        self._align = align_to_reference
        self.device = device
        
    def forward(self, x, xref):
        batch_size = x.shape[0]
        signals_count = x.shape[1]
        # Iterate over the batch dimension using indexing
        x_aligned = torch.zeros_like(x).to(self.device)
        inds = torch.zeros(batch_size, signals_count).to(self.device)
        
        for i in range(batch_size):
            for j in range(signals_count):
                x_aligned[i][j], inds[i][j] = \
                    self._align(x[i][j], xref[i][j])
        return x_aligned, inds  # Stack processed vectors
    
   
def align_to_reference(x, xref, force_copy=False):
    """
    Aligns a signal (x) to a reference signal (xref) using circular shift.
    
    Args:
        x: A numpy array of the signal to be aligned.
        xref: A numpy array of the reference signal.
    
    Returns:
        A numpy array of the aligned signal.
    """
    
    # Check if input arrays have the same size
    assert x.shape == xref.shape, "x and xref must have identical size"
    assert len(x.shape) == 1, "x shape is greater than 1 dim"
    org_shape = x.shape
    
    # Reshape to column vectors
    x = x.flatten()
    xref = xref.flatten()
    
    with torch.autocast(device_type=device_type, enabled=False):
        x = x.to(torch.float32, copy=force_copy)
        xref = xref.to(torch.float32, copy=force_copy)
        # Compute FFTs
        x_fft = torch.fft.fft(x)
        xref_fft = torch.fft.fft(xref)
        
        # Compute correlation using inverse FFT of complex conjugate product
        correlation_x_xref = torch.real(torch.fft.ifft(torch.conj(x_fft) * xref_fft))
    
    # Find index of maximum correlation
    ind = torch.argmax(correlation_x_xref).item()
    
    # Perform circular shift
    x_aligned = torch.roll(x, ind)
    
    return x_aligned.reshape(org_shape), ind
           


def align_to_reference_batched(x, xref, force_copy=False):
    """
    Aligns each signal in batch x to its corresponding xref using circular shift.
    
    Args:
        x: Tensor of shape (B, N)
        xref: Tensor of shape (B, N)
    
    Returns:
        x_aligned: Tensor of shape (B, N)
        shift_inds: LongTensor of shape (B,)
    """
    assert x.shape == xref.shape and x.ndim == 2
    B, N = x.shape

    with torch.autocast(device_type=device_type, enabled=False):
        x = x.to(torch.float32, copy=force_copy)
        xref = xref.to(torch.float32, copy=force_copy)

        x_fft = torch.fft.fft(x, dim=-1)
        xref_fft = torch.fft.fft(xref, dim=-1)
        corr = torch.real(torch.fft.ifft(torch.conj(x_fft) * xref_fft, dim=-1))

        shift_inds = torch.argmax(corr, dim=-1)  # shape (B,)

    x_aligned = torch.stack([torch.roll(x[b], shifts=int(shift_inds[b]), dims=0) for b in range(B)], dim=0)

    return x_aligned, shift_inds

def compute_cost_matrix(pred, target, bs_calc, loss_criterion='mse', fp16=False):
    """
    Compute the KxK cost matrix where each entry (i, j) is the minimal L2 distance
    between x_i and the best circularly shifted x_pred_j.
    """
    B, K, N = pred.shape
    cost_matrix = torch.zeros((B, K, K))
    
    for i in range(K):
        for j in range(K):
            pred_signal = pred[:, j, :]  # shape (B, N)
            target_signal = target[:, i, :]  # shape (B, N)
            if loss_criterion == "mse":
                aligned_pred, _ = align_to_reference_batched(pred_signal, target_signal, fp16)
                cost_matrix[:, i, j] = torch.norm(aligned_pred - target_signal, dim=1)**2 / torch.norm(target_signal, dim=1)**2
            else: # self.loss_criterion == "bs_mse"
                bs_pred, _ = bs_calc(pred_signal, "batched") # shape (B, 2, N, N)
                bs_target, _ = bs_calc(target_signal, "batched") # maybe its already calculated
                sh = bs_pred.shape
                cost_matrix[:, i, j] = torch.norm((bs_pred - bs_target).view(sh[0], -1), dim=-1)**2 / \
                    torch.norm(bs_target.view(sh[0], -1), dim=-1)**2
    
    return cost_matrix  # shape: (B, K, K)


def greedy_match(cost_matrix):
    B, K, _ = cost_matrix.shape
    matched_indices = []

    for b in range(B):
        cost = cost_matrix[b].clone()
        matched = []
        used_rows = set()
        used_cols = set()

        for _ in range(K):
            min_val = float('inf')
            min_i = min_j = -1
            for i in range(K):
                if i in used_rows: continue
                for j in range(K):
                    if j in used_cols: continue
                    if cost[i, j] < min_val:
                        min_val = cost[i, j]
                        min_i, min_j = i, j

            matched.append((min_i, min_j))
            used_rows.add(min_i)
            used_cols.add(min_j)

        matched_indices.append(matched)

    return matched_indices  # list of B lists of (i, j) tuples