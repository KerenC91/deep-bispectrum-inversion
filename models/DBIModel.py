import torch
from torch import nn
from models.model_utils import ResnetBlock, ConvBlock, set_activation, set_reduce_height
from models.network_swinir import RSTB, PatchEmbed, PatchUnEmbed


class DBIModel(nn.Module):
    """DBIModel module - same as 2, without mid layer

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """

    def __init__(self, input_len, signals_count,
                 pre_residuals,
                 pre_conv_channels,
                 post_residuals,
                 reduce_height,
                 embed_dim,
                 activation,
                 window_size,
                 patch_size,
                 depths,
                 num_heads,
                 qkv_bias,
                 qk_scale,
                 drop,
                 attn_drop,
                 drop_path_rate,
                 norm_layer,
                 downsample,
                 resi_connection,
                 use_transformers=True
                 ):
        super(DBIModel, self).__init__()

        self.use_transformers = use_transformers
        self.bs_channels = 2
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, signals_count)
        self.act_fn = set_activation(activation)
        pre_conv_channels.append(embed_dim)
        if self.use_transformers:
            self._transformers_init(depths, drop, attn_drop, input_len, patch_size, window_size, qkv_bias, qk_scale,
                                    num_heads, resi_connection, norm_layer, downsample, drop_path_rate)
        # Create pre_conv layer
        self.pre_conv = self._set_pre_conv_layers(pre_conv_channels, pre_residuals)

        if self.use_transformers:
            self.transformer_layers = self._set_transformer_layers()
            self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)

        self.reduce_height = set_reduce_height(self.embed_dim, reduce_height, input_len)

        # Create post layer - only residuals, count set by input parameter
        self.post_conv = self._set_post_conv_layers(post_residuals)

    def _transformers_init(self, depths, drop, attn_drop, img_size, patch_size, window_size, qkv_bias, qk_scale,
                           num_heads, resi_connection, norm_layer, downsample, drop_path_rate):
        self.num_layers = len(depths)
        self.drop_rate = drop
        self.drop_path_rate = drop_path_rate
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        self.attn_drop_rate = attn_drop
        self.img_size = img_size
        self.patch_norm = True
        self.patch_size = patch_size
        self.depths = depths
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.num_heads = num_heads
        self.num_features = self.embed_dim
        self.resi_connection = resi_connection
        self.norm_layer = nn.LayerNorm if norm_layer else None
        self.downsample = nn.Module if downsample else None
        self.norm = self.norm_layer(self.num_features)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size,
            in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)

        self.patches_resolution = self.patch_embed.patches_resolution
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)

    def _set_transformer_layers(self):
        # build Residual Swin Transformer blocks (RSTB)

        layers = []
        dpr = [x.item() for x in
               torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        for i_layer in range(self.num_layers):
            layer = RSTB(dim=self.embed_dim,
                         input_resolution=(
                             self.patches_resolution[0],
                             self.patches_resolution[1]),
                         depth=self.depths[i_layer],
                         num_heads=self.num_heads[i_layer],
                         window_size=self.window_size,
                         qkv_bias=self.qkv_bias,
                         qk_scale=self.qk_scale,
                         drop=self.drop_rate,
                         attn_drop=self.attn_drop_rate,
                         drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                         # no impact on SR results
                         norm_layer=self.norm_layer,
                         downsample=self.downsample,
                         # use_checkpoint=use_checkpoint,
                         img_size=self.img_size,
                         patch_size=self.patch_size,
                         resi_connection=self.resi_connection
                         )
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward_swin_transformers(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.transformer_layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def _set_pre_conv_layers(self, pre_conv_channels, pre_residuals):
        pre_convs = []

        c0 = pre_conv_channels[0]

        # add first pre_conv layer: bs_channels-->pre_conv_channels[0]
        # add residuals after layer
        pre_convs.append(ConvBlock(self.bs_channels, c0, kernel_size=3, padding=1))
        # add resnets - no change in channels dim
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))

        # add additional pre_convs layer: pre_conv_channels[i]-->pre_conv_channels[i + 1]
        # add residuals after each layer
        # pre_conv_channels set by input parameter
        for i in range(len(pre_conv_channels) - 1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        return nn.Sequential(*pre_convs)

    def _set_post_conv_layers(self, post_residuals):
        post_convs = []

        for i in range(post_residuals):
            post_convs.append(ResnetBlock(self.embed_dim, self.embed_dim, one_d=True, kernel_size=5))
        return nn.Sequential(*post_convs)

    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T # 100X100X2

        Returns:
            Tensor: B x C x (2^#channels * T) # 100X100X(2^#channels * 2)
        """
        x = self.pre_conv(x)
        if self.use_transformers:
            x = self.conv_after_body(self.forward_swin_transformers(x)) + x
        # for BXCXHXW reduce dimension to BXCX1XW
        x = self.reduce_height(x)
        x = x.squeeze(2)
        # x *= self.f
        x = self.post_conv(x)
        # x *= self.f
        x = self.linear(x.transpose(1, 2))
        x = self.act_fn(x).transpose(2, 1)

        return x
