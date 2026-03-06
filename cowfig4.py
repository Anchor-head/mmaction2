_base_ = ['configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_16xb32-u8_mitv1-rgb.py']

# load_from = 'work_dirs/cowfig/epoch_24.path'

# model settings
model = dict(
    #backbone=dict(
    #    init_cfg=None
    #)
    cls_head=dict(
        num_classes=37
    ),
    data_preprocessor=dict(
        mean=[78.84082545, 81.63554015, 86.30218749],
        std=[50.4025435, 49.69885254, 48.50408477]
    )
)
'''
# Increase total epochs
train_cfg = dict(max_epochs=50)  # Default 24

# Update the scheduler to match the new epochs
param_scheduler = [
    dict(type='LinearLR', start_factor=1/20, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min_ratio=1/20, by_epoch=True, begin=5, end=50, convert_to_iter_based=True) # Changed end to 50
]
'''

param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min_ratio=1/20, by_epoch=True, begin=0, end=24, convert_to_iter_based=True)
]

# Learning rate
optim_wrapper = dict(
    optimizer=dict(lr=2e-3),  # Default is 2e-5. Try 1e-4 or 5e-5.
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0), # Freeze backbone
            'cls_head': dict(lr_mult=1.0)  # Head learns at full speed
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0
    )
)
'''
# Batch size
train_dataloader = dict(batch_size=16) # Default is 8
'''
work_dir = './work_dirs/cowfig4'
