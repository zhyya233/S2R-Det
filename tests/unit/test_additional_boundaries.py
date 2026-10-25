import json
import numpy as np
import pytest
import torch
from torch import nn
from s2r_det.data.coco import CocoImages
from s2r_det.data.loading import raw_training_batches
from s2r_det.sampling.streams import uniform_batches, subset_support
from s2r_det.evaluation.degradations import corrupt, SETTINGS
from s2r_det.evaluation.subgroups import subgroup_transitions
from s2r_det.evaluation.matching import transitions
from s2r_det.deployment.validation import compare_outputs, parameter_count
from s2r_det.sensitivity.manifest import build_manifest
from s2r_det.training.variants import configure_variant
from s2r_det.training.protocol import RepairProtocol


@pytest.mark.parametrize("condition", list(SETTINGS))
def test_all_corruptions_are_deterministic(condition):
    image = np.arange(31 * 29 * 3, dtype=np.uint16).reshape(31, 29, 3).astype(np.uint8)
    for level in (1, 2, 3):
        a = corrupt(image, condition, level, "synthetic")
        assert a.shape == image.shape and a.dtype == np.uint8
        assert np.array_equal(a, corrupt(image, condition, level, "synthetic"))


def test_manifest_deduplicates_shared_head_and_preserves_five_groups():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.stem = nn.Conv2d(3, 3, 3, padding=1)
            self.backbone.stage3 = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1))
            self.neck = nn.Sequential(nn.Conv2d(3, 3, 3, stride=8, padding=1))
            self.bbox_head = nn.Module()
            c = nn.Conv2d(3, 3, 3, padding=1)
            self.bbox_head.cls_convs = nn.ModuleList([c, c])

        def forward(self, x, mode="tensor"):
            a = self.backbone.stage3(self.backbone.stem(x))
            self.neck(a)
            return self.bbox_head.cls_convs[0](a) + self.bbox_head.cls_convs[1](a)

    m = Model()
    manifest = build_manifest(m, torch.zeros(1, 3, 64, 64), {"head": ["bbox_head.cls_convs.0"]})
    head = [r for r in manifest if r["group"] == "head"]
    assert len(head) == 1 and len(head[0]["aliases"]) == 2 and head[0]["insertion_eligible"]
    assert any(r["group"] == "neck_p3" for r in manifest)


def test_full_gt_loader_retains_large_and_ambiguous_scope(tmp_path):
    from PIL import Image

    Image.new("RGB", (100, 50)).save(tmp_path / "one.png")
    ds = {
        "images": [{"id": 1, "file_name": "one.png", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 80, 40], "iscrowd": 0},
        ],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(ds))
    data = CocoImages(path, tmp_path)
    loaded = list(raw_training_batches(data, uniform_batches(["1"], 1, 2), 0, workers=0))
    assert len(loaded[0]) == 2 and len(loaded[0][0]["gt"]) == 2
    assert loaded[0][0]["meta"]["resized_height"] == 320


def test_recovery_damage_and_subgroup_populations():
    gt = [
        {
            "annotation_id": str(i),
            "category_id": 1,
            "box": [i * 20, 0, i * 20 + 10, 10],
            "area": 100,
            "input_area": 100,
            "occlusion": i if i < 3 else None,
        }
        for i in range(4)
    ]
    b = [{"category_id": 1, "box": gt[i]["box"]} for i in [0, 1]]
    r = [{"category_id": 1, "box": gt[i]["box"]} for i in [1, 2, 3]]
    result = transitions(gt, b, r)
    assert (result["recovered"], result["damaged"], result["net"]) == (2, 1, 1)
    assert result["success_retention_percent"] == 50
    groups = subgroup_transitions([{"gt": gt, "baseline": b, "repaired": r}], [2, 5])
    assert groups["groups"]["scale:small"]["GT"] == 4 and groups["missing_occlusion_GT"] == 1
    assert sum(v["GT"] for k, v in groups["groups"].items() if k.startswith("occlusion:")) == 3


def test_norm_scope_requires_an_explicit_set():
    model = nn.Module()
    model.conv = nn.Conv2d(96, 96, 3, padding=1)
    model.norm = nn.BatchNorm2d(96)
    with pytest.raises(ValueError):
        configure_variant(model, "flcr_plus_norm", ["conv"])


def test_numerical_agreement_is_not_an_ap_claim():
    result = compare_outputs(torch.ones(3), torch.ones(3) + 1e-5)
    assert result["within_tolerance"] and "not post-NMS/AP" in result["scope"]
    with pytest.raises(ValueError):
        compare_outputs(torch.ones(3), torch.ones(4))
    assert parameter_count(nn.Conv2d(2, 3, 3, bias=False)) == 54


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": -1}, {"steps": 0.5}, {"learning_rate": float("nan")}, {"weight_decay": -0.1}],
)
def test_invalid_protocol_settings(kwargs):
    with pytest.raises(ValueError):
        RepairProtocol(**kwargs)


def test_support_is_explicit_and_ties_are_deterministic():
    assert subset_support(["b", "a", "c"], 2, scores={"a": 1, "b": 1, "c": 0}) == ["a", "b"]
    assert subset_support(["a", "b", "c"], 2, 0) == subset_support(["a", "b", "c"], 2, 0)
    with pytest.raises(ValueError):
        subset_support(["a"], 2)


def test_fcsl_can_choose_neck_instead_of_head_and_uses_observed_q():
    from s2r_det.sensitivity.scoring import select_placement

    result = {
        "groups": {
            "head": {"q": 0.1, "top1_frequency": 0.99},
            "neck_p3": {"q": 2.0, "top1_frequency": 0.61},
        }
    }
    manifest = [
        {"path": g, "group": g, "insertion_eligible": True, "in_channels": 16, "out_channels": 16}
        for g in ["head", "neck_p3"]
    ]
    assert select_placement(result, manifest, 1000)["groups"] == ["neck_p3"]


def test_activation_hook_validation_is_atomic():
    from s2r_det.sensitivity.autograd import ActivationRecorder

    model = nn.Sequential(nn.Conv2d(1, 1, 3), nn.BatchNorm2d(1))
    with pytest.raises(TypeError):
        with ActivationRecorder(model, ["0", "1"]):
            pass
    assert not model[0]._forward_hooks
