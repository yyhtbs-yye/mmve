from mmengine.model import BaseModule
from mmve.nn.utils import flow_warp


class Warper(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat, flow):
        return flow_warp(feat, flow)