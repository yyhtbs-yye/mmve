from mmengine.model import BaseModule
from mmve.nn.utils import flow_warp


class Warper(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat_supp, flow, feat_curr=None):
        return flow_warp(feat_supp, flow)