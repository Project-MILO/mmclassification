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

data = dict(
    samples_per_gpu=128,
    train=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/train',
        ann_file='data/liveness_1fps/meta/train.txt',
        classes=classes,
        # pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/val',
        ann_file='data/liveness_1fps/meta/val.txt',
        classes=classes,
        # pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/liveness_1fps/test',
        ann_file='data/liveness_1fps/meta/test.txt',
        classes=classes,
        # pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='accuracy',
                  metric_options={'topk': (1, )})
