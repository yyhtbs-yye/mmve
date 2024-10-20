from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class SecondOrderRecurrentPropagatorNowarp(BaseModule):
    
    def __init__(self, mid_channels=64,
                 fextor_def=None, fextor_args=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        self.fextor = fextor_def(**fextor_args)

    def forward(self, curr_feats, flows):

        n, t, c, h, w = curr_feats.size()

        feat_indices = list(range(-1, -t - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(t))

        history_feats = [curr_feats[:, feat_indices[0], ...], curr_feats[:, feat_indices[0], ...]]

        out_feats = []

        for i in range(0, t):
            
            x = curr_feats[:, feat_indices[i], ...]
            y2, y1 = history_feats

            # Concatenate conditions for deformable convolution.
            c = torch.stack([y2, y1, x], dim=1)

            o = self.fextor(c) + x

            out_feats.append(o.clone())

            if i == t - 1: # for the last iter, need to to update history
                break
            # update history feats and flows
            history_feats = [history_feats[1], o]

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)
