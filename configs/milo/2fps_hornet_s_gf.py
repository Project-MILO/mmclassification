_base_ = [
    '../_base_/models/hornet/hornet-small-gf.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]


custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/kiennt54/data/models/hornet-small-gf_3rdparty_in1k_20220915-649ca492.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)

dataset_type = 'CustomDataset'
classes = ['fake', 'real']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# TODO: play with resize and crop brol

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=224),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(256, -1)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/train',
        ann_file='data/liveness_2fps/meta/train.txt',
        classes=classes,
        # pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/val',
        ann_file='data/liveness_2fps/meta/val.txt',
        classes=classes,
        # pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/liveness_2fps/test',
        ann_file='data/liveness_2fps/meta/test.txt',
        classes=classes,
        # pipeline=test_pipeline
    )
)

optimizer = dict(lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=1.0), _delete_=True)

checkpoint_config = dict(interval=50)
evaluation = dict(interval=1, metric='accuracy',
                  metric_options={'topk': (1, )}, save_best='accuracy_top-1')


lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
