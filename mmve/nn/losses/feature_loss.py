# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import torch
import torch.nn as nn
from mmengine import MMLogger
from mmengine.runner import load_checkpoint

# from mmve.models.editors.dic import LightCNN
from mmve.registry import MODELS


class LightCNNFeature(nn.Module):
    """Feature of LightCNN.

    It is used to train DICGAN.
    """

    def __init__(self) -> None:
        super().__init__()

        model = LightCNN(3)
        self.features = nn.Sequential(*list(model.features.children()))
        self.features.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
        """

        return self.features(x)

    def init_weights(self,
                     pretrained: Optional[str] = None,
                     strict: bool = True) -> None:
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@MODELS.register_module()
class LightCNNFeatureLoss(nn.Module):
    """Feature loss of DICGAN, based on LightCNN.

    Args:
        pretrained (str): Path for pretrained weights.
        loss_weight (float): Loss weight. Default: 1.0.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self,
                 pretrained: str,
                 loss_weight: float = 1.0,
                 criterion: str = 'l1') -> None:
        super().__init__()
        self.model = LightCNNFeature()
        if not isinstance(pretrained, str):
            warnings.warn('`LightCNNFeature` model in FeatureLoss ' +
                          'should be pretrained')
        self.model.init_weights(pretrained)
        self.model.eval()
        self.loss_weight = loss_weight
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("'criterion' should be 'l1' or 'mse', "
                             f'but got {criterion}')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted tensor.
            gt (Tensor): GT tensor.

        Returns:
            Tensor: Forward results.
        """

        self.model.eval()
        pred_feature = self.model(pred)
        gt_feature = self.model(gt).detach()
        feature_loss = self.criterion(pred_feature, gt_feature)

        return feature_loss * self.loss_weight

from mmengine.model import BaseModule

class LightCNN(BaseModule):
    """LightCNN discriminator with input size 128 x 128.

    It is used to train DICGAN.

    Args:
        in_channels (int): Channel number of inputs.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            MaxFeature(in_channels, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            MaxFeature(48, 48, 1, 1, 0),
            MaxFeature(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            MaxFeature(96, 96, 1, 1, 0),
            MaxFeature(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            MaxFeature(192, 192, 1, 1, 0),
            MaxFeature(192, 128, 3, 1, 1),
            MaxFeature(128, 128, 1, 1, 0),
            MaxFeature(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            MaxFeature(8 * 8 * 128, 256, filter_type='linear'),
            nn.LeakyReLU(0.2, True), nn.Linear(256, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class MaxFeature(nn.Module):
    """Conv2d or Linear layer with max feature selector.

    Generate feature maps with double channels, split them and select the max
        feature.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 1
        filter_type (str): Type of filter. Options are 'conv2d' and 'linear'.
            Default: 'conv2d'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 filter_type='conv2d'):
        super().__init__()
        self.out_channels = out_channels
        filter_type = filter_type.lower()
        if filter_type == 'conv2d':
            self.filter = nn.Conv2d(
                in_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        elif filter_type == 'linear':
            self.filter = nn.Linear(in_channels, 2 * out_channels)
        else:
            raise ValueError("'filter_type' should be 'conv2d' or 'linear', "
                             f'but got {filter_type}')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
        """

        x = self.filter(x)
        out = torch.chunk(x, chunks=2, dim=1)
        return torch.max(out[0], out[1])