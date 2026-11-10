# Attribution and dependency notices

The S²R-Det source code in this repository is distributed under the
[MIT License](LICENSE).

Dependencies are not vendored. PyTorch, NumPy, Pillow, pycocotools/COCO API,
MMDetection, MMEngine, MMCV, and optional ONNX Runtime and TensorRT retain their
respective copyrights and license terms. Preserve their applicable notices
when redistributing dependency code or binaries. No third-party model weights
or datasets are included.

The RTMDet model-scale configurations refer to the
[MMDetection 3.3.0 configurations](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/rtmdet).
The detector integration references the
[RTMDet head API](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_head.py)
and the
[Quality Focal Loss API](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/losses/gfocal_loss.py).
