import torch
from mmengine.model import BaseModule
from mmve.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class RecurrentTemporalPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, aligner=None, fextor=None, is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        self.aligner = aligner
        self.fextor = fextor

    def forward(self, feats, extras):

        n, t, c, h, w = feats.size()

        if hasattr(self.aligner, 'reset'):
            self.aligner.reset(feats, self.is_reversed)

        if hasattr(self.fextor, 'reset'):
            self.fextor.reset(feats, self.is_reversed)

        outputs = []
        output = None

        for i in range(0, t):
            
            feat, packs = self.aligner(feats, extras, i, output)
            
            output = self.fextor(feat, packs)

            outputs.append(output.clone())

        if self.is_reversed:
            outputs = outputs[::-1]

        return torch.stack(outputs, dim=1)
