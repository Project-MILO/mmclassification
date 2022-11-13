_base_ = [
    '../_base_/models/convnext/convnext-xlarge.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        # frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/kiennt54/data/models/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth',
            prefix='backbone',
            _delete_=True,
        )),
    head=dict(
        num_classes=2,
        type='LabelSmoothLoss'
    ),
)

dataset_type = 'CustomDataset'
classes = ['fake', 'real']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
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

optimizer = dict(lr=1e-4)
