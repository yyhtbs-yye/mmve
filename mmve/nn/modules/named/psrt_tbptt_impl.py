from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

from mmve.nn.modules.optical_flow.spynet import SPyNet
from mmve.nn.modules.propagators.second_order_recurrent_with_memory import SecondOrderRecurrentPropagatorWithMemory as Propagator
from mmve.nn.modules.backbones.conv2d_blocks import ResidualBlocksWithInputConv as Backbone
from mmve.nn.modules.upsamplers.conv2d_blocks import BasicVSRUpsampler as Upsampler

@MODELS.register_module()
class PSRTTbpttImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, n_frames=7, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        fextor_def = Backbone
        fextor_args_b1 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_f1 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_b2 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_f2 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)

        self.spatial_fextor = Backbone(in_channels=3, out_channels=mid_channels, num_blocks=5, ndim=5)

        # Recurrent propagators
        self.forward_propagator1  = Propagator(mid_channels, 
                                                            fextor_def=fextor_def,
                                                            fextor_args=fextor_args_f1,)

        self.backward_propagator1 = Propagator(mid_channels,  
                                                            fextor_def=fextor_def,
                                                            fextor_args=fextor_args_b1,
                                                            is_reversed=True)

        self.forward_propagator2  = Propagator(mid_channels, 
                                                            fextor_def=fextor_def,
                                                            fextor_args=fextor_args_f2,)

        self.backward_propagator2 = Propagator(mid_channels, 
                                                            fextor_def=fextor_def,
                                                            fextor_args=fextor_args_b2,
                                                            is_reversed=True)

        self.upsampler = Upsampler(in_channels=mid_channels,  out_channels=3)

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        self.history_lrs = None
        self.history_feats1 = None

    def compute_flow(self, lrs):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        forward_flows = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows

    def forward(self, lrs):

        feats_ = self.spatial_fextor(lrs)

        if self.history_lrs is not None:
            forward_flows, backward_flows = self.compute_flow(torch.cat([self.history_lrs, lrs], axis=1))
            history_forward_flows = forward_flows[:, :2, ...]
            forward_flows = forward_flows[:, 2:, ...]
            backward_flows = backward_flows[:, 2:, ...]
        else:
            forward_flows, backward_flows = self.compute_flow(lrs)
            history_forward_flows = None

        feats1 = self.forward_propagator1(feats_, forward_flows, self.history_feats1, history_forward_flows)
        
        # 2-Order TBPTT -> record the previous 2 feats
        self.history_feats1 = feats1[:, -2:, ...].detach()
        self.history_lrs = lrs[:, -2:, ...].detach()

        feats2 = self.backward_propagator1(feats1, backward_flows)

        feats3 = self.forward_propagator2(feats2, forward_flows)

        feats4 = self.backward_propagator2(feats3, backward_flows)

        return self.upsampler(lrs, feats4)

    def reset_hidden(self):
        self.history_lrs = None
        self.history_feats1 = None


if __name__ == '__main__':
    tensor_filepath = "/workspace/mmve/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor2_7_3_64_64.pt') / 100
    model = PSRTTbpttImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output = model(input_tensor)

    print(output.shape)

