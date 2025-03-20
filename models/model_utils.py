import torch
from torch import nn
import numpy as np

class ResnetBlock(nn.Module):
    """Residual Block
    Args:
        in_channels (int): number of channels in input data
        out_channels (int): number of channels in output 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, one_d=False):
        super(ResnetBlock, self).__init__()
        self.build_conv_block(in_channels, out_channels, one_d, kernel_size=kernel_size)

    def build_conv_block(self, in_channels, out_channels, one_d, kernel_size=3):
        padding = (kernel_size -1)//2
        if not one_d:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
        )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels)
            )
        else:
            self.down = None
        
        self.act = nn.ELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): B x C x T
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            residual = self.down(residual)
        return self.act(out + residual)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, one_d=False, dilation=1):
        super(ConvBlock, self).__init__()
        if not one_d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)

        self.act = nn.ELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    