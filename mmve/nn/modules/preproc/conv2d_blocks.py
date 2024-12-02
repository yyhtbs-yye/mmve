from mmengine.model import BaseModule
import torch.nn as nn

from mmve.nn.archs import ResidualBlockNoBN
from mmve.nn.utils import make_layer

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30, kernel_size=3):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )

    def forward(self, feat):

        C, H, W = feat.shape[-3:]
        PREX = feat.shape[:-3]

        if feat.ndim > 4:
            feat = feat.view(-1, C, H, W)
            feat = self.main(feat)
            feat = feat.view(*PREX, -1, H, W)
        else: # [B, C, H, W]
            feat = self.main(feat)

        return feat