import torch
import torch.nn as nn
from mmve.nn.utils import flow_warp
    
class FirstOrderAligner(nn.Module):
    
    def __init__(self) -> None:

        super().__init__()

    def reset(self, feats, is_reversed):

        device = feats.device

        n, t, c, h, w = feats.shape

        self.feat_indices = list(range(-1, -t - 1, -1)) \
                                if is_reversed \
                                    else list(range(t))
        
        self.t = t

        self.history_feats = [feats[:, self.feat_indices[0], ...]]
        self.history_flows = [torch.zeros(n, 2, h, w, device=device)]

    def forward(self, feats, extras, i, last_output):

        if i > 0:
            self.history_feats = [last_output]
            self.history_flows = [extras['flows'][:, self.feat_indices[i - 1], ...]]

        feat = feats[:, self.feat_indices[i], ...]
        y1 = self.history_feats[0]
        f1 = self.history_flows[0]
        a1 = flow_warp(y1, f1.permute(0, 2, 3, 1))

        raw     = [y1]
        aligned = [a1]
        flows   = [f1]

        dense = [it[:, i, ...] for it in extras['prev_layer_feats']] if 'prev_layer_feats' in extras and len(extras['prev_layer_feats']) > 0 else None

        return feat, {
                'aligned': aligned, 
                'raw': raw, 
                'flows': flows, 
                'dense': dense,
        }
    
class SecondOrderAligner(nn.Module):
    
    def __init__(self) -> None:

        super().__init__()

    def reset(self, feats, is_reversed):

        device = feats.device

        n, t, c, h, w = feats.shape

        self.feat_indices = list(range(-1, -t - 1, -1)) \
                                if is_reversed \
                                    else list(range(t))
        
        self.t = t

        self.history_feats = [feats[:, self.feat_indices[0], ...], feats[:, self.feat_indices[0], ...]]
        self.history_flows = [torch.zeros(n, 2, h, w, device=device), torch.zeros(n, 2, h, w, device=device)]

    def forward(self, feats, extras, i, last_output):

        if i > 0:
            self.history_feats = [self.history_feats[1], last_output]
            self.history_flows = [self.history_flows[1], extras['flows'][:, self.feat_indices[i - 1], ...]]

        feat = feats[:, self.feat_indices[i], ...]
        y2, y1 = self.history_feats
        f2, f1 = self.history_flows
        a1 = flow_warp(y1, f1.permute(0, 2, 3, 1))
        f2 = f1 + flow_warp(f2, f1.permute(0, 2, 3, 1))
        a2 = flow_warp(y2, f2.permute(0, 2, 3, 1))

        raw     = [y2, y1]
        aligned = [a2, a1]
        flows   = [f2, f1]

        dense = [it[:, i, ...] for it in extras['prev_layer_feats']] if 'prev_layer_feats' in extras and len(extras['prev_layer_feats']) > 0 else None

        return feat, {
                'aligned': aligned, 
                'raw': raw, 
                'flows': flows, 
                'dense': dense,
        }
    