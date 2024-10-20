from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

from mmve.nn.modules.optical_flow.spynet import SPyNet
from mmve.nn.modules.propagators.second_order_recurrent_nowarp import SecondOrderRecurrentPropagatorNowarp as Propagator
from mmve.nn.modules.backbones.conv2d_blocks import ResidualBlocksWithInputConv as Backbone
from mmve.nn.modules.upsamplers.conv2d_blocks import BasicVSRUpsampler as Upsampler

@MODELS.register_module()
class PSRTNowarpImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        fextor_def = Backbone
        fextor_args_b1 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_f1 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_b2 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)
        fextor_args_f2 = dict(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks, ndim=5)

        self.spatial_fextor = Backbone(in_channels=3, out_channels=mid_channels, num_blocks=5, ndim=5)

        self.forward_propagator1  = Propagator(mid_channels, 
                                               fextor_def=fextor_def, fextor_args=fextor_args_f1,)
        self.backward_propagator1 = Propagator(mid_channels, 
                                               fextor_def=fextor_def, fextor_args=fextor_args_b1, 
                                               is_reversed=True)
        self.forward_propagator2  = Propagator(mid_channels, 
                                               fextor_def=fextor_def, fextor_args=fextor_args_f2,)
        self.backward_propagator2 = Propagator(mid_channels, 
                                               fextor_def=fextor_def, fextor_args=fextor_args_b2,
                                               is_reversed=True)

        self.upsampler = Upsampler(in_channels=mid_channels,  out_channels=3)

    def forward(self, lrs):

        feats_ = self.spatial_fextor(lrs)

        feats1 = self.forward_propagator1(feats_, None)

        feats2 = self.backward_propagator1(feats1, None)

        feats3 = self.forward_propagator2(feats2, None)

        feats4 = self.backward_propagator2(feats3, None)

        return self.upsampler(lrs, feats4)


if __name__ == '__main__':
    tensor_filepath = "/workspace/mmve/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor2_7_3_64_64.pt') / 100
    model = PSRTNowarpImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output = model(input_tensor)

    print(output.shape)

