# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmve.registry import MODELS
from mmve.structures import DataSample
from mmve.utils.typing import ForwardInputs, NoiseVar, SampleList
from mmve.nn.utils import (get_valid_noise_size, get_valid_num_batches,
                     noise_sample_fn, set_requires_grad)

from mmve.nn.losses import tecogan_loss as tl

from mmve.base_models.base_gan import BaseGAN

ModelType = Union[Dict, nn.Module]

@MODELS.register_module()
class TecoGAN(BaseGAN):

    def __init__(self,
                 generator: ModelType,
                 discriminator: ModelType,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 ema_config: Optional[Dict] = None,
                 gen_auxiliary_loss_configs: dict = {},
                 disc_auxiliary_loss_configs: dict = {},
                 use_feature_match_gan: bool=False,
                 g_disc_loss_weight: float = 1.0,
                 ):
        
        super().__init__(generator=generator, discriminator=discriminator, data_preprocessor=data_preprocessor,
                         generator_steps=generator_steps, discriminator_steps=discriminator_steps,
                         ema_config=ema_config,)

        self.use_feature_match_gan = use_feature_match_gan

        if self.use_feature_match_gan:
            self.g_disc_loss = tl.GFM_Loss(g_disc_loss_weight=g_disc_loss_weight) 
        else:
            self.g_disc_loss = tl.GDiscLoss(g_disc_loss_weight=g_disc_loss_weight) 

        self.gen_auxiliary_losses = [MODELS.build(loss_config)
                                     for loss_config in gen_auxiliary_loss_configs]
        
    def _get_gen_loss(self, out_dict):
        
        losses_dict = {}

        if self.use_feature_match_gan:
            losses_dict.update(self._get_g_fm_loss(out_dict))
        else:
            losses_dict.update(self._get_g_disc_loss(out_dict))

        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            out_dict['disc_pred_fake_g'], target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.gen_auxiliary_losses is not None and len(self.gen_auxiliary_losses) > 0:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

    def _get_loss(self, losses_dict):
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var


    def _get_gen_loss(self, out_dict): 

        # gen auxiliary loss
        if self.gen_auxiliary_losses is not None and len(self.gen_auxiliary_losses) > 0:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

    def compute_flow(self, imgs):

        n, t, c, h, w = imgs.size()
        imgs_1 = imgs[:, :-1, :, :, :].reshape(-1, c, h, w)
        imgs_2 = imgs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(imgs_1, imgs_2).view(n, t - 1, 2, h, w)
        forward_flows = self.spynet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows


    def train_generator(self, inputs, targets, optimizer_wrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """

        optimizer_wrapper = optimizer_wrapper

        # Freeze the discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False
        self.discriminator.eval()

        outputs = self.generator(**inputs)

        # obtain the high quality optical flow, 
        flows = self.compute_flow(targets)

        disc_pred_fake_g = self.discriminator(outputs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_gen_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars

    def train_discriminator(self, **kwargs,
                            ) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """

        optimizer_wrapper = kwargs.pop('optimizer_wrapper')

        real_imgs, num_batches = kwargs['img'], kwargs['img'].shape[0]
        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(lqs)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_disc_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars
