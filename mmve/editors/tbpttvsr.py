# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Dict, List, Optional, Union

from mmve.base_models import BaseEditModel
from mmve.registry import MODELS
from mmve.structures import DataSample

from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper


@MODELS.register_module()
class TbpttVSR(BaseEditModel):

    def __init__(self,
                 generator,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 segment_length=5):
        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.segment_length = segment_length

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmve.nn.archs import SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.

        data_joint_batch = self.data_preprocessor(data, True)
        data_joint_batch['data_samples'] = data_joint_batch['data_samples'].gt_img

        # Split data to segs. 
        data_samples = segmentation(data_joint_batch['data_samples'], self.segment_length)
        inputs = segmentation(data_joint_batch['inputs'], self.segment_length)

        for i in range(data_samples.size(1)):

            data_batch = dict(inputs=inputs[:, i, ...],
                            data_samples=data_samples[:, i, ...],)

            with optim_wrapper.optim_context(self):
                losses = self._run_forward(data_batch, mode='loss')  # type: ignore

            parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            optim_wrapper.update_params(parsed_losses)

        self.generator.reset_hidden()

        return log_vars

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[Union[List[DataSample], torch.Tensor]] = None,
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

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def forward_train(self, inputs, data_samples=None, **kwargs):

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

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples # .gt_img

        loss = self.pixel_loss(feats, batch_gt_data)
        self.step_counter += 1

        return dict(loss=loss)

    def forward_inference(self, inputs, data_samples=None, **kwargs):

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        # feats.shape = [b, t, c, h, w]
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        gt = data_samples.gt_img[0]
        if gt is not None and gt.data.ndim == 3:
            t = feats.size(1)
            if self.check_if_mirror_extended(inputs):
                # with mirror extension
                feats = 0.5 * (feats[:, t // 4] + feats[:, -1 - t // 4])
            else:
                # without mirror extension
                feats = feats[:, t // 2]

        # create a stacked data sample
        predictions = DataSample(
            pred_img=feats.cpu(), metainfo=data_samples.metainfo)

        self.generator.reset_hidden()

        return predictions

    def forward_tensor(self,
                       inputs: torch.Tensor,
                       data_samples: Optional[Union[List[DataSample], torch.Tensor]] = None,
                       **kwargs) -> torch.Tensor:
        
        feats = self.generator(inputs, **kwargs)

        return feats

def segmentation(img_tensor, seg_length, mode='pad'):

    # img_tensor.shape = [B, T, C, H, W]
    # return' shape should be [B, T//seg_length (+1 for pad), seg_length, C, H, W]

    t_length = img_tensor.size(1)

    if mode == 'cut':
        target_length = t_length // seg_length * seg_length

        return img_tensor[:, :target_length, ...].clone()

    elif mode == 'pad':
        # Calculate the padding size
        padding_size = seg_length - (t_length % seg_length)

        # Create the padding by repeating the last frame along the temporal dimension
        additional_batches = img_tensor[:, -1:, ...].repeat(1, padding_size, 1, 1, 1)

        # Concatenate the original tensor with the padding
        padded_tensor = torch.cat([img_tensor, additional_batches], dim=1)

        # Reshape the tensor to [B, T//seg_length, seg_length, C, H, W]
        return padded_tensor.view(img_tensor.size(0), -1, seg_length, *img_tensor.shape[2:]).clone()



