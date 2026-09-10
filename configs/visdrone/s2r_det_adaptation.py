_base_ = [
    'rtmdet_tiny_baseline.py',
]

# Common to every trainable P00 variant.
#
# Prediction Retention caches dense RTMDet locations from the frozen teacher.
# Stochastic geometric transforms would destroy location correspondence.
# Therefore all P00 adaptation methods use the same deterministic geometry.

train_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
    ),
    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
    ),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114),
        ),
    ),
    dict(
        type='PackDetInputs',
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        pipeline=train_pipeline,
    ),
)
