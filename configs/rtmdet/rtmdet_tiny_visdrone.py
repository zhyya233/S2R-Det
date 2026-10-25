"""Portable tiny architecture from the existing source; local trained weights required.
Pretraining/data/runtime paths omitted. Inference policy is preserved from that
source, not claimed authenticated against the historical result checkpoint.
"""

default_scope = "mmdet"
model = {
    "backbone": {
        "act_cfg": {"inplace": True, "type": "SiLU"},
        "arch": "P5",
        "channel_attention": True,
        "deepen_factor": 0.167,
        "expand_ratio": 0.5,
        "init_cfg": None,
        "norm_cfg": {"type": "SyncBN"},
        "type": "CSPNeXt",
        "widen_factor": 0.375,
    },
    "bbox_head": {
        "act_cfg": {"inplace": True, "type": "SiLU"},
        "anchor_generator": {"offset": 0, "strides": [8, 16, 32], "type": "MlvlPointGenerator"},
        "bbox_coder": {"type": "DistancePointBBoxCoder"},
        "exp_on_reg": False,
        "feat_channels": 96,
        "in_channels": 96,
        "loss_bbox": {"loss_weight": 2.0, "type": "GIoULoss"},
        "loss_cls": {
            "beta": 2.0,
            "loss_weight": 1.0,
            "type": "QualityFocalLoss",
            "use_sigmoid": True,
        },
        "norm_cfg": {"type": "SyncBN"},
        "num_classes": 10,
        "pred_kernel_size": 1,
        "share_conv": True,
        "stacked_convs": 2,
        "type": "RTMDetSepBNHead",
        "with_objectness": False,
    },
    "data_preprocessor": {
        "batch_augments": None,
        "bgr_to_rgb": False,
        "mean": [103.53, 116.28, 123.675],
        "std": [57.375, 57.12, 58.395],
        "type": "DetDataPreprocessor",
    },
    "neck": {
        "act_cfg": {"inplace": True, "type": "SiLU"},
        "expand_ratio": 0.5,
        "in_channels": [96, 192, 384],
        "norm_cfg": {"type": "SyncBN"},
        "num_csp_blocks": 1,
        "out_channels": 96,
        "type": "CSPNeXtPAFPN",
    },
    "test_cfg": {
        "max_per_img": 300,
        "min_bbox_size": 0,
        "nms": {"iou_threshold": 0.65, "type": "nms"},
        "nms_pre": 30000,
        "score_thr": 0.001,
    },
    "train_cfg": {
        "allowed_border": -1,
        "assigner": {"topk": 13, "type": "DynamicSoftLabelAssigner"},
        "debug": False,
        "pos_weight": -1,
    },
    "type": "RTMDet",
}
