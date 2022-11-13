_base_ = [
    '../_base_/models/efficientnet_b5.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        # frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/kiennt54/data/models/efficientnet-b5_3rdparty_8xb32_in1k_20220119-e9814430.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

# dataset settings
dataset_type = 'CustomDataset'
classes = ['fake', 'real']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=456,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=456,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/train/images',
        ann_file='data/liveness_1fps/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/val/images',
        ann_file='data/liveness_1fps/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/test/images',
        ann_file='data/liveness_1fps/meta/test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='accuracy',
                  metric_options={'topk': (1, )})
