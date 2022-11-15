_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/kiennt54/data/models/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

dataset_type = 'CustomDataset'
classes = ['fake', 'real']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# TODO: play with resize and crop brol

albu_transform = dict(
    type='Albu',
    transforms=[
        dict(type='Rotate', limit=30),
        dict(type='VerticalFlip'),
        dict(
            type='OneOf',
            transforms=[
                dict(type='IAAAdditiveGaussianNoise'),
                dict(type='GaussNoise')
            ],
            p=0.2
        ),
        dict(
            type='OneOf',
            transforms=[
                dict(type='MotionBlur', p=0.2),
                dict(type='MedianBlur', blur_limit=3, p=0.1),
                dict(type='Blur', blur_limit=3, p=0.1)
            ],
            p=0.2
        ),
        dict(type='ShiftScaleRotate', shift_limit=0.0625,
             scale_limit=0.2, rotate_limit=45, p=0.2),
        dict(
            type='OneOf',
            transforms=[
                dict(type='OpticalDistortion', p=0.3),
                dict(type='GridDistortion', p=0.1),
                dict(type='IAAPiecewiseAffine', p=0.3)
            ],
            p=0.2
        ),
        dict(
            type='OneOf',
            transforms=[
                dict(type='CLAHE', clip_limit=2),
                dict(type='IAASharpen'),
                dict(type='IAAEmboss'),
                dict(type='RandomBrightnessContrast')
            ],
            p=0.3
        ),
        dict(type='HueSaturationValue', p=0.3),
    ])

# albu_transform = build_from_cfg(albu_transform, PIPELINES)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='RandomResizedCrop', size=224),
    albu_transform,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/train',
        ann_file='data/liveness_2fps/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/val',
        ann_file='data/liveness_2fps/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/test',
        ann_file='data/liveness_2fps/meta/test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)

checkpoint_config = dict(interval=50)
evaluation = dict(interval=1, metric='accuracy',
                  metric_options={'topk': (1, )}, save_best='accuracy_top-1')


lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
