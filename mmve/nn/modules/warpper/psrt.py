import torch
import torch.nn.functional as F
from mmengine.model import BaseModule


def flow_warp_avg_patch(x, flow, 
                        patch_size=8,
                        interpolation='nearest', padding_mode='zeros', align_corners=True):
    """Patch Alignment

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, 2,h, w). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    _, _, h, w = x.size()

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    flow = F.pad(flow, (0, pad_w, 0, pad_h), mode='reflect')

    hp = h + pad_h
    wp = w + pad_w

    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, hp), torch.arange(0, wp))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = F.avg_pool2d(flow, patch_size)
    flow = F.interpolate(flow, scale_factor=patch_size, mode='nearest')
    flow = flow.permute(0, 2, 3, 1)
    grid_flow = grid + flow

    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)

    output = F.grid_sample(x.float(), 
                           grid_flow, 
                           mode=interpolation, 
                           padding_mode=padding_mode, 
                           align_corners=align_corners)

    return output

class PatchWarper(BaseModule):
    def __init__(self, patch_size=8, interpolation='nearest', padding_mode='zeros', align_corners=True):
        super().__init__()
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, feat_supp, flow, feat_curr=None):
        return flow_warp_avg_patch(feat_supp, flow, patch_size=self.patch_size)