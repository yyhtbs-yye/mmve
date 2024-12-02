from mmengine.model import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from mmve.nn.modules.basics.attention_3d import SelfAttention3d
from mmve.nn.modules.basics import mlp
from mmve.nn.modules.utils import windowing, mask


class SwinTransformerBlock(nn.Module):

    def __init__(self,
                 dim, num_heads, window_size,
                 shift_size, mlp_ratio, 
                 qkv_bias, qk_scale,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)

        self.attn = SelfAttention3d(
            dim=dim, window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale
            )

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = mlp.TwoLayerMLP(in_features=dim, mid_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, attn_mask):
        b, t, h, w, c = x.shape

        shortcut = x
        
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = windowing.window_partition_3d(shifted_x,
                                     self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if any(i > 0 for i in self.shift_size):
            attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        else:
            attn_windows = self.attn(x_windows, mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, c)
        shifted_x = windowing.window_reverse_3d(attn_windows, self.window_size, b, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :t, :h, :w, :].contiguous()

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        # b,t,h,w,c
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None):

        super().__init__()
        self.dim = dim  
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0], window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale) for i in range(depth)
        ])

    def forward(self, x, attn_mask):
        #x (b,t,c,h,w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  #(b,t,h,w,c)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        #(b, t, h, w, c) -> (b, t, c, h, w)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  #b,c,t,h,w
        return x


class RSTB(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 qkv_bias, qk_scale):
        super(RSTB, self).__init__()

        self.dim = dim  
        self.residual_group = BasicLayer(
            dim=dim, depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, attn_mask):
        n, t, c, h, w = x.shape
        x_ori = x
        x = self.residual_group(x, attn_mask)
        
        x = x.view(-1, c, h, w)
        x = self.conv(x)
        x = x.view(n, t, -1, h, w)
        x = x + x_ori
        return x

    
class SwinIRFM(BaseModule):

    def __init__(self,
                 volume_size=[3, 64, 64],   # Input frames' size. Default T=3 Frames, H=64 Height, W=64 Width
                 embed_dim=96,              # Patch embedding dimension. Default: 96
                 depths=(6, 6, 6, 6),       # Depth of each Swin Transformer layer.
                 num_heads=(6, 6, 6, 6),    # Number of attention heads in different layers.
                 window_size=(3, 8, 8),     # Window size. Default: 7
                 mlp_ratio=4.,              # Ratio of mlp hidden dim to embedding dim. Default: 4
                 qkv_bias=True,             # If True, add a learnable bias to query, key, value. Default: True
                 qk_scale=None,             # Override default qk scale of head_dim ** -0.5 if set. Default: None
                 norm_layer=nn.LayerNorm,   # Normalization layer. Default: nn.LayerNorm.
                 ape=False,                 # If True, add absolute position embedding to the patch embedding. Default: False
                 patch_norm=True,           # If True, add normalization after patch embedding. Default: True
                 upscale=4,                 # Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
                 ):
        super(SwinIRFM, self).__init__()
        
        self.upscale = upscale

        self.window_size = window_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.num_frames = volume_size[0]

        self.num_layers = len(depths)  
        self.embed_dim = embed_dim  
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim  
        self.mlp_ratio = mlp_ratio  

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, volume_size[-1] * volume_size[-2], embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # Build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, now, aligned, raw=None, flows=None, dense=None):

        x = torch.stack(aligned + [now], dim=1) # -2, -1, now

        b, t, c, h, w = x.size()                        # c = embed_dim

        x_center = x[:, -1, :, :, :].contiguous()

        feats = (x + self.absolute_pos_embed) if self.ape else x
                
        attn_mask = mask.compute_mask_3d((t, h, w), tuple(self.window_size), self.shift_size).to(x.device)

        for layer in self.layers:
            feats = layer(feats.contiguous(), attn_mask)

        # b, t, c, h, w -> b, t, h, w, c
        feats = feats.permute(0, 1, 3, 4, 2).contiguous()
        
        feats = self.norm(feats)

        # b, t, h, w, c -> b, t, c, h, w
        feats = feats.permute(0, 1, 4, 2, 3).contiguous()

        return self.conv_after_body(feats[:, -1, :, :, :]) + x_center
    