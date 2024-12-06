import os

cuda_id = 0
cfg_path = "configs/n_to_n_vsr.py"

model_configs = dict(type='VanillaVSRImpl', mid_channels=32, num_blocks=7,
                     spynet_pretrained='https://download.openmmlab.com/'
                     'mmediting/restorers/basicvsr/'
                     'spynet_20210409-c6c1bd09.pth')

os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

import torch  # Assuming PyTorch is the backend
from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile(cfg_path)

cfg.train_cfg['val_interval'] = 10000

cfg.model['generator'].update(**model_configs)
# Below will not work, as they are not modified in settings but as global variables. 
cfg.train_dataloader['dataset']['num_input_frames'] = 7
cfg.val_dataloader['dataset']['num_input_frames'] = 7
cfg.work_dir = './data/work_dirs/VanillaVSRImpl'
runner = Runner.from_cfg(cfg)

runner.train()
