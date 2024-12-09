import torch
import torch.nn as nn

from mmve.nn.utils.flow_warp import flow_warp
from mmve.registry import MODELS

from mmve.nn.losses import perceptual_loss
from mmve.nn.losses.pixelwise_loss import CharbonnierLoss

import einops

@MODELS.register_module()
class TecoGANPixelLoss(CharbonnierLoss):
    def __init__(self,  **kwargs):
        self.pixel_loss_weight = kwargs.pop("pixel_loss_weight")

        super().__init__(**kwargs)

    def forward(self, **kwargs):
        sr, gt = kwargs['sr'], kwargs['gt']

        return self.pixel_loss_weight * super().forward(sr, gt)
    
    def loss_name(self) -> str:
        return "pixel_loss"

@MODELS.register_module()
class TecoGANPerceptualLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vgg_loss = perceptual_loss.PerceptualLoss(**kwargs).to('cuda')

    def forward(self, **kwargs):
        sr, gt = kwargs['sr'], kwargs['gt']

        sr = einops.rearrange(sr, 'b t c h w -> (b t) c h w')
        gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')

        percep_loss, _ = self.vgg_loss(sr, gt)
        return percep_loss

    def loss_name(self) -> str:
        return "perceptual_loss"

@MODELS.register_module()
class TecoGANWarpingRegularization(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.warping_reg_weight = kwargs.pop('warping_reg_weight')
        self.warping_reg = nn.MSELoss()


    def forward(self, **kwargs):

        lq, lq_forward_flows = kwargs['lq'], kwargs['lq_forward_flows']
        lq_tp1 = lq[:, 1:, ...]
        lq_t = lq[:, :-1, ...]

        lq_tp1 = einops.rearrange(lq_tp1, 'b t c h w -> (b t) c h w')
        lq_t = einops.rearrange(lq_t, 'b t c h w -> (b t) c h w')
        lq_forward_flows = einops.rearrange(lq_forward_flows, 'b t c h w -> (b t) c h w')

        lq_tp1_warped = flow_warp(lq_t, lq_forward_flows.permute(0, 2, 3, 1))
        reg = self.warping_reg_weight * self.warping_reg(lq_tp1, lq_tp1_warped)
        return reg

    def loss_name(self) -> str:
        return "warping_regularization"

@MODELS.register_module()
class TecoGANPingpongRegularization(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pingpong_reg_weight = kwargs.pop('pingpong_reg_weight')
        self.pingpong_reg = nn.L1Loss()  # Assuming L1 Loss for simplicity

    def forward(self, **kwargs):
        sr = kwargs['sr']
        n_frames = sr.size(1) // 2  #  13 -> n_frames=6
        sr_forward = sr[:, :n_frames, ...]              # 0,  1,  2,  3,  4,  5
        sr_backward = sr[:, n_frames+1:, ...].flip(1)   # 12, 11, 10, 9,  8,  7
        return self.pingpong_reg_weight * self.pingpong_reg(sr_forward, sr_backward)

    def loss_name(self) -> str:
        return "pingpong_regularization"

@MODELS.register_module()
class TecoGANGDiscLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.g_disc_loss_weight = kwargs.pop('g_disc_loss_weight')
        self.cls_loss = nn.BCELoss() 

    def forward(self, **kwargs):


        sr = kwargs['sr'][:, :-1, ...]
        hq_forward_flows = kwargs['hq_forward_flows']

        b = sr.size(0)

        reshaped_sr = einops.rearrange(sr, 'b t c h w -> (b t) c h w')
        reshaped_hq_forward_flows = einops.rearrange(hq_forward_flows, 'b t c h w -> (b t) c h w')

        discriminator = kwargs['discriminator']

        sr_warped = flow_warp(reshaped_sr, reshaped_hq_forward_flows.permute(0, 2, 3, 1))

        sr_warped = einops.rearrange(sr_warped, '(b t) c h w -> b t c h w', b=b)

        sr_cls = discriminator(sr, sr_warped)  # Assuming a discriminator that takes two inputs
        loss = self.g_disc_loss_weight * self.cls_loss(sr_cls, torch.ones_like(sr_cls))
        return loss

    def loss_name(self) -> str:
        return "g_disc_loss"

@MODELS.register_module()
class TecoGANGFM_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.g_disc_loss_weight = kwargs.pop('g_disc_loss_weight')
        self.feature_loss = nn.L1Loss()  # Assuming L1 Loss for feature comparison

    def forward(self, **kwargs):

        sr, gt = kwargs['sr'], kwargs['gt']

        sr = sr[:, :-1, ...]
        gt = gt[:, :-1, ...]

        reshaped_sr = einops.rearrange(sr, 'b t c h w -> (b t) c h w')
        reshaped_gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')

        hq_forward_flows = kwargs['hq_forward_flows']

        hq_forward_flows = einops.rearrange(hq_forward_flows, 'b t c h w -> (b t) c h w')

        discriminator = kwargs['discriminator']

        sr_warped = flow_warp(sr, hq_forward_flows.permute(0, 2, 3, 1))
        gt_warped = flow_warp(gt, hq_forward_flows.permute(0, 2, 3, 1))

        sr_feats = discriminator(sr, sr_warped, get_features=True)
        gt_feats = discriminator(gt, gt_warped, get_features=True)
        loss = self.g_disc_loss_weight * self.feature_loss(sr_feats, gt_feats)
        return loss

    def loss_name(self) -> str:
        return "g_fm_loss"


@MODELS.register_module()
class TecoGANDLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.d_teco_loss_weight = kwargs.pop('d_teco_loss_weight')
        self.cls_loss = nn.BCELoss()  

    def forward(self, **kwargs):
        sr, gt = kwargs['sr'], kwargs['gt']

        b = sr.size(0)

        sr = sr[:, :-1, ...]
        gt = gt[:, :-1, ...]
        
        reshaped_sr = einops.rearrange(sr, 'b t c h w -> (b t) c h w')
        reshaped_gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')

        hq_forward_flows = kwargs['hq_forward_flows']

        reshaped_hq_forward_flows = einops.rearrange(hq_forward_flows, 'b t c h w -> (b t) c h w')

        discriminator = kwargs['discriminator']

        sr_warped = flow_warp(reshaped_sr, reshaped_hq_forward_flows.permute(0, 2, 3, 1))
        gt_warped = flow_warp(reshaped_gt, reshaped_hq_forward_flows.permute(0, 2, 3, 1))

        sr_warped = einops.rearrange(sr_warped, '(b t) c h w -> b t c h w', b=b)
        gt_warped = einops.rearrange(gt_warped, '(b t) c h w -> b t c h w', b=b)

        sr_cls = discriminator(sr, sr_warped)
        gt_cls = discriminator(gt, gt_warped)

        sr_loss = self.cls_loss(sr_cls, torch.zeros_like(sr_cls))
        gt_loss = self.cls_loss(gt_cls, torch.ones_like(gt_cls))
        loss = self.d_teco_loss_weight * (sr_loss + gt_loss)
        return loss

    def loss_name(self) -> str:
        return "d_teco_loss"
