from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

from mmve.nn.modules.optical_flow.spynet import  SPyNet
from mmve.nn.modules.propagators.rtp import RecurrentTemporalPropagator as Propagator
from mmve.nn.modules.aligners.deform_conv_align import SecondOrderDeformableAligner as Aligner
from mmve.nn.modules.backbones.conv2d_blocks import ResidualBlocksWithInputConv as Fextor
from mmve.nn.modules.preproc.conv2d_blocks import ResidualBlocksWithInputConv as Preproc
from mmve.nn.modules.upsamplers.conv2d_blocks import BasicVSRUpsampler

@MODELS.register_module()
class BananaVSRImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        
        fextor_b1, fextor_f1, fextor_b2, fextor_f2 = [
            Fextor(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks) 
            for _ in range(4)
        ]

        aligner_b1, aligner_f1, aligner_b2, aligner_f2 = [
            Aligner(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1) 
            for _ in range(4)
        ]

        self.spatial_fextor = Preproc(3, mid_channels, 5)

        # Recurrent propagators
        self.backward_propagator1 = Propagator(mid_channels, aligner=aligner_b1, fextor=fextor_b1, is_reversed=True)
        self.forward_propagator1  = Propagator(mid_channels, aligner=aligner_f1, fextor=fextor_f1,)
        self.backward_propagator2 = Propagator(mid_channels, aligner=aligner_b2, fextor=fextor_b2, is_reversed=True)
        self.forward_propagator2  = Propagator(mid_channels, aligner=aligner_f2, fextor=fextor_f2,)
        
        self.upsampler = BasicVSRUpsampler(mid_channels, mid_channels, 3)

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained)

    # def compute_flow(self, lrs):

    #     n, t, c, h, w = lrs.size()
    #     lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
    #     lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

    #     backward_flows = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
    #     forward_flows = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

    #     return forward_flows, backward_flows

    def forward(self, lrs):

        n, t, c, h, w = lrs.size()

        feats_ = self.spatial_fextor(lrs)

        # compute optical flow
        # forward_flows, backward_flows = self.compute_flow(lrs)

        feats1 = self.backward_propagator1(feats_, extras={})#, extras={'flows': backward_flows})

        feats2 = self.forward_propagator1(feats1, extras={})#, extras={'flows': forward_flows})

        feats3 = self.backward_propagator2(feats2, extras={})#, extras={'flows': backward_flows})
        
        feats4 = self.forward_propagator2(feats3, extras={})#, extras={'flows': forward_flows})

        return self.upsampler(feats4, lrs)

if __name__ == '__main__':
    tensor_filepath = "/workspace/mmve/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor2_7_3_64_64.pt') / 100
    model = BananaVSRImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output1 = model(input_tensor)

