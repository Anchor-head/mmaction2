_base_ = ['configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_16xb32-u8_mitv1-rgb.py']

load_from = 'ckpts/uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_16xb32-u8_mitv1-rgb_20230313-a6f4a567.pth'

# model settings
model = dict(
    backbone=dict(
        init_cfg=None
    ),
    cls_head=dict(
        num_classes=10
    ),
    data_preprocessor=dict(
        mean=[76.80265384, 78.65608052, 83.88651252],
        std=[50.23659895, 49.05482588, 48.27881695]
    )
)

# Increase total epochs
train_cfg = dict(max_epochs=200)  # Default 24

# Update the scheduler to match the new epochs
param_scheduler = [
    dict(type='LinearLR', start_factor=1/20, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min_ratio=1/20, by_epoch=True, begin=5, end=200, convert_to_iter_based=True)
]



# Learning rate
base_lr = 1e-4  # Default is 2e-5. Try 1e-4 or 5e-5.
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # Only micro-adjust backbone
            'cls_head': dict(lr_mult=1.0)  # Head learns at full speed
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0
    )
)

# Redefine training pipeline
num_frames = 8
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    # dict(type='RandomResizedCrop'), # Disabled
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    batch_size=16) # Default is 8

work_dir = './work_dirs/cowfig-premiers10-uncropped'
