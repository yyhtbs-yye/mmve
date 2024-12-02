from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

from mmve.nn.modules.optical_flow.spynet import  SPyNet
from mmve.nn.modules.propagators.second_order_recurrent_dense_net import SecondOrderRecurrentPropagatorDenseNet as Propagator
from mmve.nn.modules.backbones.smsrt_block import SwinIRFM 
from mmve.nn.modules.preproc.conv2d_blocks import ResidualBlocksWithInputConv as preproc
from mmve.nn.modules.upsamplers.conv2d_blocks import BasicVSRUpsampler

@MODELS.register_module()
class SwinVsr3DImpl(BaseModule):

    def __init__(self, mid_channels=32, volume_size=[3, 64, 64], depths=(6, 6, 6), num_heads=(4, 4, 4),
                 spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        
        fextor_b1 = SwinIRFM(volume_size=volume_size, embed_dim=mid_channels, depths=depths, num_heads=num_heads)
        fextor_f1 = SwinIRFM(volume_size=volume_size, embed_dim=mid_channels, depths=depths, num_heads=num_heads)
        fextor_b2 = SwinIRFM(volume_size=volume_size, embed_dim=mid_channels, depths=depths, num_heads=num_heads)
        fextor_f2 = SwinIRFM(volume_size=volume_size, embed_dim=mid_channels, depths=depths, num_heads=num_heads)

        self.spatial_fextor = preproc(3, mid_channels, 5)

        # Recurrent propagators
        self.backward_propagator1 = Propagator(mid_channels, fextor=fextor_b1, is_reversed=True)
        self.forward_propagator1  = Propagator(mid_channels, fextor=fextor_f1,)
        self.backward_propagator2 = Propagator(mid_channels, fextor=fextor_b2, is_reversed=True)
        self.forward_propagator2  = Propagator(mid_channels, fextor=fextor_f2,)
        
        self.upsampler = BasicVSRUpsampler(mid_channels, mid_channels, 3)

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

    def compute_flow(self, lrs):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        forward_flows = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows

    def forward(self, lrs):

        n, t, c, h, w = lrs.size()

        feats_ = self.spatial_fextor(lrs)

        # compute optical flow
        forward_flows, backward_flows = self.compute_flow(lrs)

        feats1 = self.backward_propagator1(feats_, backward_flows, None)

        feats2 = self.forward_propagator1(feats1, forward_flows, None)

        feats3 = self.backward_propagator2(feats2, backward_flows, None)

        feats4 = self.forward_propagator2(feats3, forward_flows, None)

        return self.upsampler(feats4, lrs)

if __name__ == '__main__':
    tensor_filepath = "/workspace/mmve/test_input_tensor2_3_3_64_64.pt"
    input_tensor = torch.load('test_input_tensor2_3_3_64_64.pt') / 100
    model = SwinVsr3DImpl(mid_channels=32, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth').to('cuda:2')
    
    model.eval()

    output1 = model(input_tensor.to('cuda:2'))

