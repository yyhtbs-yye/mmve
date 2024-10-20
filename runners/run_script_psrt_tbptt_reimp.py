
import os

cuda_id = 2
cfg_path = "configs/n_to_n_vsr.py"

num_input_frames = 25
segment_length = 5
model_configs = dict(type='PSRTTbpttImpl', mid_channels=32, num_blocks=15, n_frames=num_input_frames,
                     spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

import torch  # Assuming PyTorch is the backend
from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile(cfg_path)

cfg.model['generator'].update(**model_configs)
cfg.model['type'] = 'TbpttVSR'
cfg.model['segment_length'] = segment_length

# Below will not work, as they are not modified in settings but as global variables. 
cfg.train_dataloader['dataset']['num_input_frames'] = num_input_frames
cfg.val_dataloader['dataset']['num_input_frames'] = num_input_frames
cfg.work_dir = './work_dirs/PSRTTbpttImpl'
runner = Runner.from_cfg(cfg)

runner.train()
