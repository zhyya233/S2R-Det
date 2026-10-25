import math
import pytest
import torch
from torch import nn
from s2r_det.adapters.flcr import (
    FLCRConv2d,
    insert_flcr,
    fuse_flcr_inplace,
    adapter_parameter_count,
)
from s2r_det.failure_atlas.atlas import preliminary, build_atlas, image_pools, trace_nms
from s2r_det.sampling.streams import bounded_failure_quotas, FailureStratifiedBatchSource
from s2r_det.sensitivity.scoring import group_logs, bootstrap_stability, select_placement
from s2r_det.sensitivity.cohorts import build_cohorts, density_thresholds, density_label
from s2r_det.sensitivity.autograd import ActivationRecorder, diagnostic_loss
from s2r_det.retention.cache import build_cache_entry
from s2r_det.retention.losses import prediction_retention_loss, pooled_retention_loss
from s2r_det.evaluation.matching import greedy_match
from s2r_det.evaluation.statistics import seed_summary, paired_sequence_bootstrap, replicate_coco
from s2r_det.training.protocol import learning_rate_factor, calibration_choice
from s2r_det.training.variants import configure_variant
from s2r_det.utils.io import require_identity, fingerprint

torch.set_num_threads(1)


@pytest.mark.parametrize(
    "stride,dilation,padding,mode",
    [
        (1, 1, 1, "zeros"),
        (2, 1, 1, "zeros"),
        (1, 2, 2, "zeros"),
        (1, 1, 1, "reflect"),
        (1, 1, 1, "replicate"),
        (1, 1, 1, "circular"),
        (1, 1, 0, "zeros"),
    ],
)
def test_zero_init_freeze_shapes_and_fusion(stride, dilation, padding, mode):
    torch.manual_seed(8)
    base = nn.Conv2d(
        3, 5, 3, stride=stride, dilation=dilation, padding=padding, padding_mode=mode, bias=True
    )
    layer = FLCRConv2d(base, 2)
    x = torch.randn(2, 3, 11, 13)
    assert torch.equal(layer(x), base(x))
    assert not any(p.requires_grad for p in base.parameters())
    with torch.no_grad():
        layer.repair_u.weight.normal_(std=0.15)
    assert layer(x).shape == base(x).shape
    torch.testing.assert_close(layer(x), layer.fused_conv()(x), rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "base,rank",
    [
        (nn.Conv2d(4, 4, 1), 8),
        (nn.Conv2d(4, 4, 3, groups=2), 8),
        (nn.Conv2d(4, 4, 3), 0),
        (nn.Conv2d(4, 4, 3), 1.5),
    ],
)
def test_unsupported_flcr(base, rank):
    with pytest.raises(ValueError):
        FLCRConv2d(base, rank)


def test_shared_head_count_and_fused_aliases():
    model = nn.Module()
    conv1 = nn.Conv2d(96, 96, 3, padding=1)
    conv2 = nn.Conv2d(96, 96, 3, padding=1)
    model.cls = nn.ModuleList([conv1, conv1, conv1])
    model.reg = nn.ModuleList([conv2, conv2, conv2])
    insert_flcr(model, ["cls.0", "reg.0"], 8)
    assert adapter_parameter_count(model) == 3216
    assert model.cls[0] is model.cls[1] is model.cls[2]
    fuse_flcr_inplace(model)
    assert model.cls[0] is model.cls[2]
    assert not any(isinstance(m, FLCRConv2d) for m in model.modules())


def gt(box=(0, 0, 10, 10), identity="g", cls=1):
    return {
        "annotation_id": identity,
        "box": list(box),
        "category_id": cls,
        "area": 100.0,
        "input_area": 100.0,
    }


def pred(box=(0, 0, 10, 10), score=0.8, cls=1):
    return {"box": list(box), "score": score, "category_id": cls}


def trace(post=(), suppressed=(), capped=()):
    return {
        "post_indices": list(post),
        "suppressed_indices": list(suppressed),
        "capped_indices": list(capped),
    }


@pytest.mark.parametrize(
    "prediction,expected", [(None, "F1"), (pred((0, 0, 2, 10)), "F2"), (pred(), "success_pending")]
)
def test_atlas_basic_categories(prediction, expected):
    p = [] if prediction is None else [prediction]
    assert preliminary([gt()], p, trace(range(len(p))))[0]["category"] == expected


def test_f3_quartile_equality_and_f5():
    images = [{"image_id": "a", "gt": [gt()], "pre": [pred(score=0.5)], "trace": trace([0])}]
    assert build_atlas(images, {"1:vt16": 0.5})["records"][0]["category"] == "F3"
    images[0]["pre"][0]["score"] = 0.6
    assert build_atlas(images, {"1:vt16": 0.5})["records"][0]["category"] == "F5"


def test_cap_is_not_confirmed_nms():
    candidates = [pred(), pred((20, 20, 30, 30), 0.9)]
    capped = preliminary([gt()], candidates, trace([1], capped=[0]))[0]
    suppressed = preliminary([gt()], candidates, trace([1], suppressed=[0]))[0]
    assert capped["category"] == "ambiguous" and suppressed["category"] == "F4"


def test_duplicate_assignment_and_class_consistency():
    rows = preliminary([gt(identity="a"), gt(identity="b")], [pred()], trace([0]))
    assert [r["category"] for r in rows] == ["success_pending", "ambiguous"]
    assert greedy_match([gt()], [pred(cls=2)]) == {}
    assert greedy_match([gt((0, 0, 2, 2))], [pred((0, 0, 1, 2))]) == {0: 0}


def test_pool_overlap_ambiguous_and_identity():
    rows = [
        {**gt(identity=str(i)), "image_id": "seq/a", "category": c}
        for i, c in enumerate(["F1", "F5", "F5", "ambiguous"])
    ]
    pools, counts, retention = image_pools(rows)
    assert pools["F1"] == retention == ["seq/a"] and counts == {"F1": 1}
    with pytest.raises(ValueError):
        image_pools(rows + [rows[0]])
    assert trace_nms([pred(), pred(), pred((30, 30, 40, 40))], 0.5, 1) == trace([0], [1], [2])


@pytest.mark.parametrize(
    "counts",
    [
        {"F1": 1, "F2": 1, "F3": 1, "F4": 1000},
        {"F1": 100, "F2": 100, "F3": 1, "F4": 1},
        {"F1": 1, "F2": 4, "F3": 7},
    ],
)
def test_bounded_quotas(counts):
    q = bounded_failure_quotas(counts)
    assert sum(q.values()) == pytest.approx(1) and all(
        0.1 - 1e-10 <= v <= 0.4 + 1e-10 for v in q.values()
    )


@pytest.mark.parametrize("counts", [{}, {"F1": 1}, {"F1": 2, "F2": 2}])
def test_infeasible_quota_population(counts):
    with pytest.raises(ValueError):
        bounded_failure_quotas(counts)


def test_sampler_half_up_and_determinism():
    q = {k: 0.25 for k in ["F1", "F2", "F3", "F4"]}
    pools = {k: [k + "-image"] for k in q}
    source = FailureStratifiedBatchSource(pools, ["retention"], q, 16, 0, 0.5)
    assert source.source_schedule(5).count("failure") == 3
    assert list(source.iter_batches(5)) == list(source.iter_batches(5))
    assert all(len(b["image_ids"]) == 16 for b in source.iter_batches(5))
    with pytest.raises(ValueError):
        FailureStratifiedBatchSource(pools, [], q)


def test_even_module_median_and_finite_validation():
    modules = [{"path": str(i), "group": "head", "analysis_eligible": True} for i in range(4)]
    assert group_logs({str(i): math.exp(v) for i, v in enumerate([0, 2, 6, 20])}, modules)[
        "head"
    ] == pytest.approx(4)
    with pytest.raises(ValueError):
        group_logs({"0": float("nan"), "1": 1, "2": 1, "3": 1}, modules)


def test_cohort_instances_not_image_cohort_labels():
    records = []
    cohorts = {
        "a": {"cohort": "failure_rich", "density": "low"},
        "b": {"cohort": "success_rich", "density": "high"},
    }
    for image, f, s in [("a", 10, 4), ("b", 6, 0)]:
        for category, value in [("F1", f), ("F5", s)]:
            records.append(
                {
                    "image_id": image,
                    "category": category,
                    "h": {"head": value, "neck_p3": value / 2},
                }
            )
    result = bootstrap_stability(records, cohorts, 20)
    assert result["groups"]["head"]["q"] == 6
    modules = [
        {
            "path": "head",
            "group": "head",
            "insertion_eligible": True,
            "in_channels": 96,
            "out_channels": 96,
        }
    ]
    assert select_placement(result, modules, 1608)["groups"] == ["head"]
    with pytest.raises(ValueError):
        select_placement(result, modules, 1)
    with pytest.raises(ValueError):
        select_placement(result, modules, 3216, number_groups=2)


def test_cohort_construction_and_training_only_density():
    records = []
    counts = {str(i): i % 5 for i in range(20)}
    for i in range(20):
        records.append({"image_id": str(i), "category": "F1" if i < 10 else "F5"})
    c = build_cohorts(records, counts, 4)
    assert len(c["images"]) == 8
    assert sum(x["cohort"] == "failure_rich" for x in c["images"].values()) == 4
    with pytest.raises(ValueError):
        build_cohorts(records, counts, 256)
    t = density_thresholds({"a": 0, "b": 3, "c": 6})
    assert density_label(t[0], t) == "low"


def test_shared_activation_element_pooling():
    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1))
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.ones(8, requires_grad=True) * 3
    with ActivationRecorder(model, ["0"]) as recorder:
        recorder.calls["0"] = [a, b]
        value = recorder.sensitivities(2 * a.sum() + b.sum() * 0)["0"]
    assert value == pytest.approx((6 / 10) / (math.sqrt((5 + 72) / 10) + 1e-12))


def test_diagnostic_proxy_max_iou_and_no_assigner():
    class Head:
        def loss_cls(self, logits, target, weight, avg_factor):
            assert target[0].item() == 1 and target[1].item() == 1 and avg_factor == 1
            return logits.sum()

        def loss_bbox(self, boxes, target, weight, avg_factor):
            return (boxes - target).square().sum()

    logits = torch.zeros(2, 2, requires_grad=True)
    boxes = torch.tensor([[20.0, 20, 30, 30], [0, 0, 10, 10]], requires_grad=True)
    loss, index = diagnostic_loss(Head(), logits, boxes, torch.tensor([0.0, 0, 10, 10]), 1)
    assert index == 1
    loss.backward()
    assert logits.grad[0].sum() == 0


def test_raw_cache_repetition_empty_and_no_f5():
    logits = torch.tensor([[-100.0, -100.0], [3.0, 2.0], [float("nan"), 1.0]])
    boxes = torch.tensor([[0.0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]])
    cache = build_cache_entry("i", logits, boxes, [], 100)
    assert len(cache["dense_indices"]) == 4 and cache["dense_indices"].count(1) == 2
    assert not any(cache["box_mask"])
    empty = build_cache_entry("i", torch.empty(0, 2), torch.empty(0, 4), [])
    assert empty["teacher_logits"] == []


def test_retention_detached_teacher_distinct_beta_delta():
    student = torch.zeros(2, 2, requires_grad=True)
    teacher = torch.ones(2, 2, requires_grad=True)
    boxes = torch.full((2, 4), 2.0, requires_grad=True)
    target = torch.zeros(2, 4, requires_grad=True)
    loss = prediction_retention_loss(
        student, boxes, teacher, target, torch.tensor([True, False]), beta=2.0, delta=1.0
    )
    assert loss["box"].item() == pytest.approx(1.5)
    assert loss["total"].item() == pytest.approx(loss["classification"].item() + 3)
    loss["total"].backward()
    assert teacher.grad is None and target.grad is None and boxes.grad[1].abs().sum() == 0
    empty = prediction_retention_loss(
        torch.empty(0, 2),
        torch.empty(0, 4),
        torch.empty(0, 2),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.bool),
    )
    assert empty["total"].item() == 0


def test_pooled_retention_is_not_mean_of_image_means():
    def pair(n, value):
        return (
            torch.full((n, 1), value),
            torch.zeros(n, 4),
            torch.zeros(n, 1),
            torch.zeros(n, 4),
            torch.zeros(n, dtype=torch.bool),
        )

    a, b = pair(1, 0.0), pair(3, 3.0)
    pooled = pooled_retention_loss([a, b], torch.tensor(0.0))["total"]
    individual = [prediction_retention_loss(*p)["total"] for p in [a, b]]
    assert pooled.item() == pytest.approx((individual[0] + 3 * individual[1]).item() / 4)


def test_schedule_endpoints_and_calibration_ties():
    assert learning_rate_factor(0, 42525) == pytest.approx(1e-5)
    assert learning_rate_factor(999, 42525) == 1 and learning_rate_factor(1000, 42525) == 1
    assert learning_rate_factor(42524, 42525) == pytest.approx(0.05)
    records = [
        {"steps": 12150, "split": "validation", "APsmall": 2, "AP": 3, "learning_rate": v}
        for v in [0.01, 0.001]
    ]
    assert calibration_choice(records)["learning_rate"] == 0.001


def test_full_finetuning_preserves_running_states():
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2))
    configure_variant(model, "full_finetuning")
    assert all(p.requires_grad for p in model.parameters()) and not model[1].training


def test_statistics_and_replica_identity():
    assert seed_summary([1, 2, 3])["sample_SD"] == 1
    with pytest.raises(ValueError):
        seed_summary([1])
    x = paired_sequence_bootstrap({"a": 1, "b": 2}, {"a": 2, "b": 3}, 100)
    assert x["CI95"] == [1.0, 1.0]
    ds = {"images": [{"id": 1}], "annotations": [{"id": 1, "image_id": 1}]}
    sample, predictions = replicate_coco(ds, [{"image_id": 1}], ["1", "1"])
    assert [i["id"] for i in sample["images"]] == [1, 2] and len(
        {a["id"] for a in sample["annotations"]}
    ) == 2


def test_cache_identity_must_match():
    identity = {
        k: fingerprint(k)
        for k in ["baseline", "annotations", "preprocessing", "dense_convention", "atlas"]
    }
    require_identity(identity, identity)
    with pytest.raises(ValueError):
        require_identity(identity, {**identity, "baseline": fingerprint("other")})
