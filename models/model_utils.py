from torch import nn

class ResnetBlock(nn.Module):
    """Residual Block
    Args:
        in_channels (int): number of channels in input data
        out_channels (int): number of channels in output 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, one_d=False):
        super(ResnetBlock, self).__init__()
        self.act = nn.ELU()
        padding = (kernel_size -1)//2
        if not one_d:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d
        # Set conv layers
        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
        )
        # Set down
        if in_channels != out_channels:
            self.down = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels)
            )
        else:
            self.down = None

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


def set_activation(activation_name):
    # ['ELU', 'LeakyReLU', 'ReLU', 'Softsign', 'Tanh'])

    if activation_name == 'ELU':
        activation = nn.ELU()
    elif activation_name == 'ReLU':
        activation = nn.ReLU()
    elif activation_name == 'Softsign':
        activation = nn.Softsign()
    elif activation_name == 'Tanh':
        activation = nn.Tanh()
    else:  # 'LeakyReLU':
        activation = nn.LeakyReLU()

    return activation


def update_reduce_height_cnt(reduce_height, Hin):
    """
    Calculate the number of layers for reducing height, based on k, s

    Parameters
    ----------
    k : int
        height kernel size.
    s : int
        height stride size.
    Hin : int
        height of the input signal.

    Returns
    -------
    cnt : int
        Number of reduce_height layers to perform.
    k : int
        height kernel size for each layer.
    s : int
        height stride size for each layer.

    """
    cnt, k, s = reduce_height

    assert Hin > k, f"Error! Hin={Hin} is smaller or equal to k={k}"

    H = Hin
    cnt = 0
    add_conv_2 = False
    while H > 1:
        H = int((H - k) / s) + 1
        cnt += 1
        if H == 2:
            add_conv_2 = True
            break
    print(f'reduce_height=[{cnt},{k},{s},{add_conv_2}]')

    return cnt, k, s, add_conv_2


def set_reduce_height(embed_dim, reduce_height, input_len):
    reduce_height = update_reduce_height_cnt(reduce_height, input_len)
    # Create pre middle layer - reduce height
    reduce_height_layers = []
    cnt, k, s, add_conv_2 = reduce_height

    for _ in range(cnt):
        reduce_height_layers.append(nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                              kernel_size=(k, 1), stride=(s, 1)))
    if add_conv_2:
        reduce_height_layers.append(nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                                              kernel_size=(2, 1), stride=(2, 1)))
    return nn.Sequential(*reduce_height_layers)