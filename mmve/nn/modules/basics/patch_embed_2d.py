import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, num_frames=5, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.num_frames = num_frames

        self.in_chans = in_chans  
        self.embed_dim = embed_dim  

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # Input size of x: (n, t, c, h, w)
        # n: batch size
        # t: number of frames (temporal dimension)
        # c: number of channels
        # h: height of each frame
        # w: width of each frame

        n, t, c, h, w = x.size() 

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # After permute: (n, c, t, h, w)
        # Rearranges dimensions so that channels (c) come first, followed by frames (t), height (h), and width (w).

        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            # flatten(2): Collapses dimensions (t, h, w) into a single dimension.
            # Result size: (n, c, t*h*w)
            # transpose(1, 2): Swaps the channel dimension (c) and the flattened spatial dimension (t*h*w).
            # Result size: (n, t*h*w, c)

            x = self.norm(x)

            # Applies normalization to the last dimension (channels).
            # Result size (unchanged): (n, t*h*w, c)

            x = x.transpose(1, 2).view(-1, self.embed_dim, t, h, w)
            # transpose(1, 2): Swaps dimensions back, so channels (c) are last again.
            # Result size before view: (n, c, t*h*w)
            # view reshapes tensor to (-1, embed_dim, t, h, w), where embed_dim is the normalized feature size.
            # Result size: (n, embed_dim, t, h, w)

        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x
