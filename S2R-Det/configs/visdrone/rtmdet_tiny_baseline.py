_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

data_root = 'data/links/visdrone2019_det/'

classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
)

metainfo = dict(classes=classes)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
        )
    ),
    bbox_head=dict(num_classes=10)
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='coco_annotations/visdrone2019_det_train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='coco_annotations/visdrone2019_det_val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        metainfo=metainfo
    )
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='coco_annotations/visdrone2019_det_testdev.json',
        data_prefix=dict(img='VisDrone2019-DET-test-dev/images/'),
        metainfo=metainfo
    )
)

val_evaluator = dict(
    ann_file=data_root + 'coco_annotations/visdrone2019_det_val.json'
)

test_evaluator = dict(
    ann_file=data_root + 'coco_annotations/visdrone2019_det_testdev.json'
)

# 单 GPU按官方总 batch=256 线性缩放 LR
auto_scale_lr = dict(enable=True, base_batch_size=256)

randomness = dict(seed=0, deterministic=False)

# 控制磁盘；正式 baseline 最多保留最近 checkpoint + best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=2,
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

work_dir = 'outputs/b00_baseline'

# Frozen single-GPU optimization policy.
# Official RTMDet-tiny reference total batch size is 256.
# S2R-Det uses one GPU with batch_size=16, therefore:
# 0.004 * 16 / 256 = 0.00025.
auto_scale_lr = dict(enable=False, base_batch_size=256)

optim_wrapper = dict(
    optimizer=dict(lr=0.00025)
)

# Frozen scaled scheduler policy.
# Official RTMDet schedule uses eta_min = base_lr * 0.05.
# After scaling 0.004 -> 0.00025 for total batch 16,
# eta_min is scaled identically: 0.00025 * 0.05 = 0.0000125.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0000125,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True),
]
