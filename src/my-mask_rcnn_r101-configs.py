_base_ = [
    '/home/trainer/JennyHo/Visual_HW3/mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py',
    '/home/trainer/JennyHo/Visual_HW3/mmdetection/configs/_base_/datasets/coco_instance.py',
    '/home/trainer/JennyHo/Visual_HW3/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/home/trainer/JennyHo/Visual_HW3/mmdetection/configs/_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'CocoDataset'

classes = ('nuclei', )  # need a comma at the end if there is only one class

# resize image because cuda out of memory
train_pipeline = [
    dict(type='Resize', img_scale=(600, 800), keep_ratio=True),
]

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101'),
        frozen_stages=1,),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8],
            ratios=[0.5, 1.0],)
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1),
        ),
    train_cfg=dict(gpu_assign_thr=50),
)

data = dict(
    samples_per_gpu=1,  # decrease batch size because cuda out of memory
    frozen_stages=1,
    train=dict(
        type=dataset_type,
        ann_file='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/annotations/train.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/images/train/'),
    val=dict(
        type=dataset_type,
        ann_file='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/annotations/val.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/images/train/'),
    test=dict(
        type=dataset_type,
        ann_file='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/annotations/test.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_HW3/mmdetection/coco/images/test/'))

# schedule settings
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=400)

load_from = '/home/trainer/JennyHo/Visual_HW3/mmdetection/checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'
