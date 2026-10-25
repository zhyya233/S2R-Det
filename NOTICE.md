# Attribution and dependency notices

FLCR and Conv-Adapter primitives derive from the author-supplied S²R-Det source.
The package-level software license has not yet been selected; see [LICENSE](LICENSE).

Dependencies are not vendored. PyTorch, NumPy, Pillow, pycocotools/COCO API,
MMDetection, MMEngine, MMCV, and optional ONNX Runtime and TensorRT retain their
respective copyrights and license terms. Preserve their applicable notices when
redistributing dependency code or binaries. No third-party model weights or
datasets are included.

The RTMDet scale templates follow the architecture dimensions in the
[MMDetection 3.3.0 configurations](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/rtmdet).
The native detector adapter follows the
[RTMDet head API](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_head.py)
and [Quality Focal Loss API](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/losses/gfocal_loss.py).
These references identify integration provenance and do not establish historical
experiment equivalence or redistribution clearance.
