import importlib.util
import pytest
import torch
from torch import nn
from s2r_det.training.engine import repair
from s2r_det.training.protocol import RepairProtocol
from s2r_det.sampling.streams import uniform_batches
from s2r_det.adapters.flcr import FLCRConv2d


def test_toy_cpu_repair_preserves_frozen_parameters():
    class Toy:
        def __init__(self):
            self.model = nn.Sequential(
                FLCRConv2d(nn.Conv2d(1, 1, 3, padding=1), 1), nn.BatchNorm2d(1)
            )

        def prepare(self, dataset, image_ids):
            return torch.ones(len(image_ids), 1, 4, 4)

        def detection_loss(self, inputs):
            return {"loss_det": self.model(inputs).square().mean()}

    toy = Toy()
    toy.model[1].requires_grad_(False)
    result = repair(
        toy,
        None,
        uniform_batches(["toy"], 4, 2),
        RepairProtocol(steps=4, batch_size=2, warmup_steps=2, retention_lambda=0),
    )
    assert result["completed_steps"] == 4 and result["frozen_state_verified"]


def test_coco_and_strict_vt16_boundary():
    pytest.importorskip("pycocotools")
    from s2r_det.evaluation.coco import standard_metrics

    ds = {
        "images": [{"id": 1, "width": 640, "height": 640}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 16, 16],
                "area": 256,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [30, 30, 15, 15],
                "area": 225,
                "iscrowd": 0,
            },
        ],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [30, 30, 15, 15], "score": 0.9}]
    m = standard_metrics(ds, predictions)
    assert m["VT16 AP@500"] == pytest.approx(100) and m["APsmall"] < 100
    assert standard_metrics(ds, [])["AP"] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("mmdet") is None,
    reason="MMDetection/MMCV unavailable; no detector/checkpoint integration claim",
)
def test_mmdetection_import_boundary():
    from mmdet.models.dense_heads import RTMDetSepBNHead

    assert RTMDetSepBNHead is not None


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="ONNX Runtime optional backend unavailable",
)
def test_onnx_dependency():
    import onnxruntime

    assert onnxruntime.__version__
