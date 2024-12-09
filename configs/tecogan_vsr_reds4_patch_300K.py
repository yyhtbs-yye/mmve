_base_ = [
    './_base_/default_runtime.py',
]

model_name = 'TecoGAN_frvsr'
model_configs = dict()

train_iter = 300_000
val_interval = 100
train_on_patch = True
gt_patch_size = 256
work_dir = f'./data/work_dirs/{model_name}'
save_dir = './data/work_dirs'
scale = 4

# ----------- Train Data Delivery
batch_size = 1
num_workers = 15

num_input_frames = 7

# model settings
model = dict(
    type='TecoGAN',
    generator=dict(
        type='UniBasicVSRImpl', mid_channels=32, num_blocks=7,
        spynet_pretrained='https://download.openmmlab.com/'
        'mmediting/restorers/basicvsr/'
        'spynet_20210409-c6c1bd09.pth',
    ),
    discriminator=dict(
        type='SpatioTemporalDiscriminator', in_channels=3, mid_channels=32,
    ),
    gen_losses=[
        dict(
            type='TecoGANPixelLoss',
            pixel_loss_weight=1,
        ),
        dict(
            type='TecoGANPerceptualLoss',
            vgg_type='vgg16',
            layer_weights={
                '4': 0.2,
                '9': 0.2,
                '16': 0.2,
            },
            perceptual_weight=0.05,
            style_weight=120,
            pretrained=('torchvision://vgg16'),
        ),
        dict(
            type='TecoGANWarpingRegularization',
            warping_reg_weight=1,
        ),
        dict(
            type='TecoGANPingpongRegularization',
            pingpong_reg_weight=0.5,
        ),
        dict(
            type='TecoGANGDiscLoss',
            g_disc_loss_weight=0.01,
        ),
    ],
    disc_losses=[
        dict(
            type='TecoGANDLoss',
            d_teco_loss_weight=1,
        ),
    ],
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    *( [dict(type='PairedRandomCrop', gt_patch_size=gt_patch_size)] if train_on_patch else [] ),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='MirrorSequence', keys=['img', 'gt']),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PackInputs')
]

reds_root = './data/REDS'

train_dataloader = dict(
    num_workers=num_workers,
    batch_size=batch_size,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=reds_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_train.txt',
        depth=1,
        num_input_frames=num_input_frames,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=num_workers,
    batch_size=batch_size,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=reds_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=1,
        num_input_frames=num_input_frames,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(type='IterBasedTrainLoop', max_iters=train_iter, val_interval=val_interval)
val_cfg = dict(type='MultiValLoop')

paramwise_cfg = dict(custom_keys={'spynet': dict(lr_mult=0.125)})

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',

    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5, betas=(0.9, 0.999)),
    ),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5, betas=(0.9, 0.990)),
    )
)

# learning policy
param_scheduler = dict(type='CosineRestartLR', by_epoch=False,
                       periods=[300_000], restart_weights=[1], eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,
        save_optimizer=True,
        out_dir=save_dir,
        by_epoch=False))