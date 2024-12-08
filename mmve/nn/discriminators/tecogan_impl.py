import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmve.registry import MODELS

@MODELS.register_module()
class DiscriminatorBlocks(nn.Module):
    def __init__(self, in_channels=64, norm_layer=nn.BatchNorm2d):
        super(DiscriminatorBlocks, self).__init__()

        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(in_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.block5 = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, padding=0, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pools to a 1x1 feature map
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, get_features=False):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        feature_list = [h1, h2, h3, h4]

        if get_features:
            return feature_list
        else:
            return self.fc(self.global_pool(self.block5(h4)))

@MODELS.register_module()
class SpatioTemporalDiscriminator(BaseModule):

    def __init__(self, in_channels=64):
        super(SpatioTemporalDiscriminator, self).__init__()

        # input conv.
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.main = DiscriminatorBlocks(in_channels=in_channels)  # downsample 16x


    def forward(self, x, x_warped, get_features=False):

        # NOW:
        # x.shape = [n, t, c, h, w]
        # x_warped.shape = [n, t, c, h, w]

        x = torch.cat([x[:, i:i + 3] for i in range(x.shape[1] - 2)], dim=0)
        x_warped = torch.cat([x_warped[:, i:i + 3] for i in range(x_warped.shape[1] - 2)], dim=0)

        # NOW: after sliding window
        # x.shape = [n*(t-1), 3*c, h, w]
        # x_warped.shape = [n*(t-1), 3*c, h, w]

        h = self.preproc(torch.cat([x, x_warped], dim=1))

        return self.main(h, get_features=get_features)

