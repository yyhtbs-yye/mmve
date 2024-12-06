import torch
import torch.nn as nn

import functools

from mmengine.model import BaseModule
from mmve.registry import MODELS
import torch.nn.functional as F

class BicubicUpsampler(nn.Module):

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsampler, self).__init__()

        # calculate weights (according to Eq.(6) in the reference paper)
        cubic = torch.FloatTensor([
            [0, a, -2*a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2*a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s**2, s**3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))  # size: (f, 4)

    def forward(self, input):
        n, c, h, w = input.size()
        f = self.scale_factor

        # merge n&c
        input = input.reshape(n*c, 1, h, w)

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (vertical expansion)
        kernel_h = self.kernels.view(f, 1, 4, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0)
        output = output.permute(0, 2, 1, 3).reshape(n*c, 1, f*h, w + 3)

        # calculate output (horizontal expansion)
        kernel_w = self.kernels.view(f, 1, 1, 4)
        output = F.conv2d(output, kernel_w, stride=1, padding=0)
        output = output.permute(0, 2, 3, 1).reshape(n*c, 1, f*h, f*w)

        # split n&c
        output = output.reshape(n, c, f*h, f*w)

        return output

class DiscriminatorBlocks(nn.Module):
    def __init__(self):
        super(DiscriminatorBlocks, self).__init__()

        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        feature_list = [out1, out2, out3, out4]

        return out4, feature_list

def get_upsampling_func(scale=4, degradation='BI'):
    if degradation == 'BI':
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode='bilinear',
            align_corners=False)

    elif degradation == 'BD':
        upsample_func = BicubicUpsampler(scale_factor=scale)

    else:
        raise ValueError(f'Unrecognized degradation type: {degradation}')

    return upsample_func

def backward_warp(x, flow, mode='bilinear', padding_mode='border'):

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


@MODELS.register_module()
class SpatioTemporalDiscriminator(BaseModule):
    """ Spatio-Temporal discriminator proposed in TecoGAN
    """

    def __init__(self, in_nc, spatial_size, tempo_range, degradation, scale):
        super(SpatioTemporalDiscriminator, self).__init__()

        # basic settings
        mult = 3  # (conditional triplet, input triplet, warped triplet)
        self.spatial_size = spatial_size
        self.tempo_range = tempo_range
        assert self.tempo_range == 3, 'currently only support 3 as tempo_range'
        self.scale = scale

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # downsample 16x

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale, degradation)

    def forward(self, lqs, hqs, sqs, hq_flows, generator):

        n, t, c, lr_h, lr_w = lqs.size()
        _, _, _, hr_h, hr_w = hqs.size()

        s_size = self.spatial_size
        t = t // 3 * 3  # discard other frames
        n_clip = n * t // 3  # total number of 3-frame clips in all batches

        c_size = int(s_size * args_dict['crop_border_ratio'])
        n_pad = (s_size - c_size) // 2

        # === compute forward & backward flow === #
        hq_forward_flows, hq_backward_flows = hq_flows
        hr_flow_idle = torch.zeros_like(hr_flow_bw)
        # merge bw/idle/fw flows
        hr_flow_merge = torch.stack(
            [hr_flow_bw, hr_flow_idle, hr_flow_fw], dim=2)  # n,t//3,3,2,h,w

        # reshape and stop gradient propagation
        hr_flow_merge = hr_flow_merge.view(n_clip * 3, 2, hr_h, hr_w).detach()

        # === build up inputs for D (3 parts) === #
        # part 1: bicubic upsampled hqs (conditional inputs)
        cond_data = sqs[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        # note: permutation is not necessarily needed here, it's just to keep
        #       the same impl. as TecoGAN-Tensorflow (i.e., rrrgggbbb)
        cond_data = cond_data.permute(0, 2, 1, 3, 4)
        cond_data = cond_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 2: original hqs
        orig_data = hqs[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        orig_data = orig_data.permute(0, 2, 1, 3, 4)
        orig_data = orig_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 3: warped hqs
        warp_data = backward_warp(
            hqs[:, :t, ...].reshape(n * t, c, hr_h, hr_w), hr_flow_merge)
        warp_data = warp_data.view(n_clip, 3, c, hr_h, hr_w)
        warp_data = warp_data.permute(0, 2, 1, 3, 4)
        warp_data = warp_data.reshape(n_clip, c * 3, hr_h, hr_w)
        # remove border to increase training stability as proposed in TecoGAN
        warp_data = F.pad(
            warp_data[..., n_pad: n_pad + c_size, n_pad: n_pad + c_size],
            (n_pad,) * 4, mode='constant')

        # combine 3 parts together
        input_data = torch.cat([orig_data, warp_data, cond_data], dim=1)

        # === classify === #
        out = self.conv_in(input_data)
        out, feature_list = self.discriminator_block(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        pred = out, feature_list

        # construct output dict (return other hqs beside pred)
        ret_dict = {
            'hr_flow_merge': hr_flow_merge
        }

        return pred, ret_dict

