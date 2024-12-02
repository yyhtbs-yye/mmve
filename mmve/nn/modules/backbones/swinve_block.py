from mmengine.model import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from mmve.nn.modules.basics.attention_3d import WinMHSelfAttention3d
from mmve.nn.modules.basics import mlp
from mmve.nn.modules.utils import windowing, mask

class SwinTransformerBlock(nn.Module):

    def __init__(self,
                 dim, num_heads, 
                 window_size, shift_size,
                 mlp_ratio, 
                 qkv_bias, 
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.attn = WinMHSelfAttention3d(
            dim=dim, num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = mlp.TwoLayerMLP(in_features=dim, mid_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, attn_mask):
        B, T, H, W, C = x.shape
        
        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_z = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_z = x

        # partition windows
        h_windows = windowing.window_partition_3d(shifted_z,
                                     self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if any(i > 0 for i in self.shift_size):
            attn_windows = self.attn(h_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        else:
            attn_windows = self.attn(h_windows, mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_z = windowing.window_reverse_3d(attn_windows, self.window_size, B, T, H, W)

        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            z = torch.roll(
                shifted_z, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            z = shifted_z

        # FFN
        x = x + self.mlp(z + x)

        return x

class RSTB(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 qkv_bias):
        super().__init__()
        
        self.dim = dim  
        self.depth = depth
        
        # Define the blocks (similar to BasicLayer)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0], window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias) for i in range(depth)
        ])

    def forward(self, x, attn_mask):

        h = x.contiguous()

        # Apply each block in the residual group
        for blk in self.blocks:
            h = blk(h, attn_mask)

        # Add residual connection
        return h + x  # Residual connection


    
class SwinIRFM(BaseModule):

    def __init__(self,
                 volume_size=[3, 64, 64],   # Input frames' size. Default T=3 Frames, H=64 Height, W=64 Width
                 embed_dim=96,              # Patch embedding dimension. Default: 96
                 depths=(6, 6, 6, 6),       # Depth of each Swin Transformer layer.
                 num_heads=(6, 6, 6, 6),    # Number of attention heads in different layers.
                 window_size=(3, 8, 8),     # Window size. Default: 7
                 mlp_ratio=4.,              # Ratio of mlp hidden dim to embedding dim. Default: 4
                 qkv_bias=True,             # If True, add a learnable bias to query, key, value. Default: True
                 ):
        super(SwinIRFM, self).__init__()
        
        self.window_size = window_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.num_frames = volume_size[0]
        self.height = volume_size[1]
        self.width = volume_size[2]

        self.num_layers = len(depths)  
        self.embed_dim = embed_dim  
        self.num_features = embed_dim  
        self.mlp_ratio = mlp_ratio  


        # Build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias)
            self.layers.append(layer)

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

        feats = torch.stack(aligned + [now], dim=1) # -2, -1, now

        x_center = feats[:, -1, :, :, :].contiguous()

        b, t, c, h, w = feats.size()                        # c = embed_dim

        # print(f"t={t}, h={h}, w={w}")

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]

        # This is [B, T, C, H, W] padding would be [W, H, C, T, B]
        feats = F.pad(feats, (pad_l, pad_r, pad_t, pad_b, 0, 0, pad_d0, pad_d1))

        b, t, c, h, w = feats.size()                        # c = embed_dim
                
        attn_mask = mask.compute_mask_3d((t, h, w), tuple(self.window_size), self.shift_size).to(feats.device)

        feats = feats.permute(0, 1, 3, 4, 2)

        for layer in self.layers:
            feats = layer(feats, attn_mask)
        
        feats = feats.permute(0, 1, 4, 2, 3)

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            feats = feats[:, :t-pad_d1, :, :h-pad_b, :w-pad_r].contiguous()

        return feats[:, -1, :, :, :] + x_center
    