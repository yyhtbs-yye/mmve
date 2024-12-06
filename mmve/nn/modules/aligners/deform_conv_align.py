import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init

class SecondOrderDeformableAligner(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):

        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAligner, self).__init__(*args, **kwargs)

        # Populate the layers according to the order list
        self.conv_offset1 = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),  
        ) # 27 = 9(#x_offset) + 9(#y_offset) +  9(#mask)

        # Populate the layers according to the order list
        self.conv_offset2 = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),  
        ) # 27 = 9(#x_offset) + 9(#y_offset) +  9(#mask)

        self.init_offset()

    def init_offset(self): # Init constant offset.

        constant_init(self.conv_offset1[-1], val=0, bias=0)
        constant_init(self.conv_offset2[-1], val=0, bias=0)

    def reset(self, feats, is_reversed): # Reset function.

        t = feats.size(1)

        self.feat_indices = list(range(-1, -t - 1, -1)) \
                                if is_reversed \
                                    else list(range(t))

        self.history_feats = [feats[:, self.feat_indices[0], ...], feats[:, self.feat_indices[0], ...]]

    def forward(self, feats, extras, i, last_output):

        if i > 0:
            self.history_feats = [self.history_feats[1], last_output]

        feat = feats[:, self.feat_indices[i], ...]

        y2, y1 = self.history_feats

        i1 = torch.cat([y1, feat], dim=1)
        i2 = torch.cat([y2, feat], dim=1)

        p1 = self.conv_offset1(i1)
        o1, m1 = torch.split(p1, [18 * self.deform_groups, 9 * self.deform_groups], dim=1)

        p2 = self.conv_offset2(i2)
        o2, m2 = torch.split(p2, [18 * self.deform_groups, 9 * self.deform_groups], dim=1)

        # offset
        o1 = self.max_residue_magnitude * torch.tanh(o1)
        o2 = self.max_residue_magnitude * torch.tanh(o2)
        # mask
        m1 = torch.sigmoid(m1)
        m2 = torch.sigmoid(m2)

        a1 = modulated_deform_conv2d(y1, o1, m1, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)

        a2 = modulated_deform_conv2d(y2, o2, m2, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)

        raw     = [y2, y1]
        aligned = [a2, a1]

        dense = [it[:, i, ...] for it in extras['prev_layer_feats']] if 'prev_layer_feats' in extras and len(extras['prev_layer_feats']) > 0 else None

        return feat, {
            'aligned': aligned, 
            'raw': raw,
            'flows': None, 
            'dense': dense,

        }

