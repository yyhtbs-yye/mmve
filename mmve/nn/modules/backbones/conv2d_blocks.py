from mmengine.model import BaseModule
import torch.nn as nn

from mmve.nn.archs import ResidualBlockNoBN
from mmve.nn.utils import make_layer

import einops

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30, ndim=4, kernel_size=3):
        super().__init__()

        if ndim == 4:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
            )
        elif ndim == 5: # For the case of 3 inputs frames
            self.main = nn.Sequential(
                nn.Conv2d(in_channels*3, out_channels, kernel_size, 1, kernel_size//2, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
            )

        self.ndim = ndim

    def forward(self, feat, aux=None, dense=None):
        if self.ndim == 5:
            feat = einops.rearrange(feat, 'b t c h w -> b (t c) h w')

        return self.main(feat)