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
        if method == "sum":
            source = torch.zeros(batch_size, self.targets_count, self.channels, self.height, self.width).to(self.device)
      
            for i in range(batch_size):
                for j in range(self.targets_count):
                    source[i][j], target[i][j] = self._create_data(target[i][j])
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
    
   
def align_to_reference(x, xref):
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
        x = x.to(torch.float32)
        xref = xref.to(torch.float32)
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
           


