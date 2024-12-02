from mmengine.model import BaseModule
import torch.nn as nn
import torch
from mmve.nn.archs import ResidualBlockNoBN
from mmve.nn.utils import make_layer

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30, kernel_size=3):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2, bias=True),
            nn.GELU(),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )
    
    def forward(self, now, aligned, raw, flows, dense=None):

        feat = torch.cat(aligned + [now], dim=1) # -2, -1, now
            
        if dense is None:
            return self.main(feat)
        
        dense = torch.cat(dense, dim=1)

        return self.main(torch.concat([feat, dense], 1))