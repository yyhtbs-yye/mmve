from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class TemporalIndependentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64,
                 fextor_def=None, fextor_args=None,):

        super().__init__()

        self.mid_channels = mid_channels

        self.fextor = fextor_def(**fextor_args)

    def forward(self, curr_feats):

        out_feats = []

        for i in range(0, curr_feats.size(1)):
            
            x = curr_feats[:, i:i+1, ...]

            o = self.fextor(x) + x
            
            out_feats.append(o.clone())

        return torch.stack(out_feats, dim=1)
