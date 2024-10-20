from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmve.registry import MODELS

from mmve.nn.modules.optical_flow.spynet import  SPyNet
from mmve.nn.modules.refiner.fgd_refiner import AnyOrderDeformableAlignment as refiner
from mmve.nn.modules.feature_extractor.conv_module import ResidualBlocksWithInputConv as Extractor
from mmve.nn.modules.temporal_propagator.recurrent import FirstOrderUnidirectionalRecurrentPropagator as Propagator

from mmve.nn.modules.upsamplers.conv2d_blocks import BasicVSRUpsampler
from mmve.nn.modules.spatial_processor.conv_module import BasicVSRPlusPlusSpatial

@MODELS.register_module()
class BasicVSRPlusImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None, max_residue_magnitude=10):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        self.preproc = BasicVSRPlusPlusSpatial(in_channels=3, mid_channels=mid_channels)

        self.fextor_l1_args = dict(in_channels=2*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l2_args = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l3_args = dict(in_channels=4*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l4_args = dict(in_channels=5*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)

        self.refiner_args = dict(in_channels=mid_channels, out_channels=mid_channels, 
                                 kernel_size=3, padding=1, deform_groups=16,
                                 max_residue_magnitude=max_residue_magnitude, order=1)

        self.propagator_l1 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l1_args,
                                        refiner_def=refiner, refiner_args=self.refiner_args, is_reversed=True)
        self.propagator_l2 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l2_args,
                                        refiner_def=refiner, refiner_args=self.refiner_args)
        self.propagator_l3 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l3_args,
                                        refiner_def=refiner, refiner_args=self.refiner_args, is_reversed=True)
        self.propagator_l4 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l4_args,
                                        refiner_def=refiner, refiner_args=self.refiner_args)

        self.upsample = BasicVSRUpsampler(in_channels=5*mid_channels, mid_channels=64)

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

        # compute optical flow
        flows = self.compute_flow(lrs)

        feats_ = self.preproc(lrs)

        feats_l1 = self.propagator_l1(feats_, flows, [])
        feats_l2 = self.propagator_l2(feats_l1, flows, [feats_])
        feats_l3 = self.propagator_l3(feats_l2, flows, [feats_, feats_l1])
        feats_l4 = self.propagator_l4(feats_l3, flows, [feats_, feats_l1, feats_l2])

        return self.upsample(torch.cat([feats_, feats_l1, feats_l2, feats_l3, feats_l4], dim=-3), lrs)
