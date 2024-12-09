# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union
from copy import deepcopy

import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmve.registry import MODELS
from mmve.structures import DataSample
from mmve.nn.utils import (get_valid_noise_size, get_valid_num_batches,
                     noise_sample_fn, set_requires_grad)

from mmve.nn.losses import tecogan_loss as tgl

from mmve.base_models.base_gan import BaseGAN

ModelType = Union[Dict, nn.Module]

@MODELS.register_module()
class TecoGAN(BaseGAN):

    def __init__(self,
                 generator: dict, discriminator: dict,
                 data_preprocessor: dict,
                 generator_steps: int = 1, discriminator_steps: int = 1,
                 gen_losses: Optional[Dict] = None, disc_losses: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 ema_config: Optional[Dict] = None,
                 ):
                
        BaseModel.__init__(self, data_preprocessor) # BaseModel is the grandparent class
        self.generator = MODELS.build(generator)
        self.discriminator = MODELS.build(discriminator)

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps
        self.is_weight_fixed = False

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

        self.gen_losses = [MODELS.build(loss_config)
                                     for loss_config in gen_losses]
        
        self.disc_losses = [MODELS.build(loss_config)
                                     for loss_config in disc_losses]

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0

    def train_step(self, data, optim_wrapper):

        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        loss_dict = super().train_step(data, optim_wrapper)
        self.step_counter += 1

        return loss_dict

    def _get_gen_loss(self, out_dict):
        
        losses_dict = {}

        # gen auxiliary loss
        if self.gen_losses is not None and len(self.gen_losses) > 0:
            for loss_module in self.gen_losses:
                loss_ = loss_module(**out_dict)
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

    def _get_disc_loss(self, out_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}

        # disc auxiliary loss
        if self.disc_losses is not None and len(self.disc_losses) > 0:
            for loss_module in self.disc_losses:
                loss_ = loss_module(**out_dict)
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

    def compute_forward_flow(self, spynet, imgs):

        n, t, c, h, w = imgs.size()
        imgs_1 = imgs[:, :-1, :, :, :].reshape(-1, c, h, w)
        imgs_2 = imgs[:, 1:, :, :, :].reshape(-1, c, h, w)

        forward_flows = spynet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)

        return forward_flows

    def train_generator(self, inputs, data_samples, optimizer_wrapper) -> Dict[str, Tensor]:
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

        # Freeze the discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False
        self.discriminator.eval()

        if isinstance(inputs, dict):
            inputs = inputs['img']

        if not torch.allclose(inputs[:, 0, ...], inputs[:, -1, ...], atol=1e-6):
            raise ValueError("The video sequence is not mirrored.")

        outputs = self.generator(inputs)

        batch_gt_data = data_samples.gt_img

        spynet = self.generator.spynet

        # obtain the high quality optical flow, 
        hq_forward_flows = self.compute_forward_flow(spynet, batch_gt_data)
        lq_forward_flows = self.compute_forward_flow(spynet, inputs) # This is a problem

        data_dict_ = dict(
            gen=self.generator,
            discriminator=self.discriminator,
            sr=outputs, gt=batch_gt_data, lq=inputs,
            hq_forward_flows=hq_forward_flows,
            lq_forward_flows=lq_forward_flows,
            # iteration=curr_iter,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        
        loss, log_vars = self._get_gen_loss(data_dict_)

        optimizer_wrapper.update_params(loss)

        for param in self.discriminator.parameters():
            param.requires_grad = True
        self.discriminator.train()

        return log_vars

    def train_discriminator(self, inputs, data_samples, optimizer_wrapper) -> Dict[str, Tensor]:
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

        # Freeze the discriminator
        requires_grad_records = []
        for i, param in enumerate(self.generator.parameters()):
            requires_grad_records.append(param.requires_grad)
            
            param.requires_grad = False

        self.generator.eval()

        if isinstance(inputs, dict):
            inputs = inputs['img']

        outputs = self.generator(inputs)

        batch_gt_data = data_samples.gt_img

        spynet = self.generator.spynet

        # obtain the high quality optical flow, 
        hq_forward_flows = self.compute_forward_flow(spynet, batch_gt_data)
        lq_forward_flows = self.compute_forward_flow(spynet, inputs) # This is a problem

        data_dict_ = dict(
            gen=self.generator,
            discriminator=self.discriminator,
            sr=outputs, gt=batch_gt_data, lq=inputs,
            hq_forward_flows=hq_forward_flows,
            lq_forward_flows=lq_forward_flows,
            # iteration=curr_iter,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))

        loss, log_vars = self._get_disc_loss(data_dict_)

        optimizer_wrapper.update_params(loss)

        for i, param in enumerate(self.generator.parameters()):
            param.requires_grad = requires_grad_records[i]
        self.generator.train()

        return log_vars

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
        
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

    def forward_tensor(self,
                       inputs: torch.Tensor,
                       data_samples: Optional[List[DataSample]] = None,
                       **kwargs) -> torch.Tensor:
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(inputs, **kwargs)

        return feats


    def forward_inference(self,
                          inputs: torch.Tensor,
                          data_samples: Optional[List[DataSample]] = None,
                          **kwargs) -> DataSample:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            DataSample: predictions.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # create a stacked data sample here
        predictions = DataSample(pred_img=feats.cpu())

        return predictions
    
    def convert_to_datasample(self, predictions: DataSample,
                              data_samples: DataSample,
                              inputs: Optional[torch.Tensor]
                              ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples
