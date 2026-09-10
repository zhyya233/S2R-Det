#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from mmdet.apis import (
    inference_detector,
    init_detector,
)
from mmdet.utils import register_all_modules

from s2r_det.adapters.flcr import (
    FLCRConv2d,
    fuse_flcr_inplace,
)
from s2r_det.training.adaptation_variants import (
    configure_variant,
)


ROOT = Path(
    "/home/a/projects/S2R-Det"
)

CFG = (
    ROOT
    / "configs/base/"
      "rtmdet_tiny_visdrone_640.py"
)

BASELINE = (
    ROOT
    / "outputs/b00_baseline_corrected/"
      "best_coco_bbox_mAP_epoch_297.pth"
)

KSTAR = (
    ROOT
    / "outputs/k00/K00-01/train/last.pth"
)

KSTAR_CONTROLS = (
    ROOT
    / "experiments/manifests/k00/K00-01.json"
)

ANN = Path(
    "/home/a/projects/datasets/"
    "VisDrone2019-DET/"
    "coco_annotations/"
    "visdrone2019_det_testdev.json"
)

TEST_ROOT = Path(
    "/home/a/projects/datasets/"
    "VisDrone2019-DET/"
    "VisDrone2019-DET-test-dev"
)

RAW_ANN = (
    TEST_ROOT
    / "annotations"
)

X01X02 = (
    ROOT
    / "outputs/x00/X01-X02"
)

U00_EVIDENCE = (
    ROOT
    / "outputs/u00_flcr/"
      "final_formal_exactmetric/"
      "u00_final_gate.json"
)

EXPECTED = {
    "baseline":
        "f0e78b061985f4f33003170afa9d6c74b21301a3f751afe62edbe552a81cf475",
    "kstar":
        "d8d52b702b2282064678297789440bccdb6cf74b1270015c11b8c5876879b3e3",
}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def image_path(im):
    name = str(
        im["file_name"]
    )

    p = (
        TEST_ROOT
        / "images"
        / name
    )

    if p.is_file():
        return p

    p = TEST_ROOT / name

    if p.is_file():
        return p

    raise FileNotFoundError(
        name
    )


def load_state(
    model,
    checkpoint,
):
    x = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    state = x.get(
        "state_dict",
        x,
    )

    inc = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        inc.missing_keys
        or inc.unexpected_keys
    ):
        raise RuntimeError(
            "strict checkpoint load failed"
        )


def build_baseline():
    model = init_detector(
        str(CFG),
        str(BASELINE),
        device="cuda:0",
    )

    model.eval()

    return model


def build_kstar(
    fused=False,
):
    model = init_detector(
        str(CFG),
        str(BASELINE),
        device="cuda:0",
    )

    controls = load_json(
        KSTAR_CONTROLS
    )

    configure_variant(
        model,
        "m00_head_r8",
        controls,
    )

    load_state(
        model,
        KSTAR,
    )

    model.eval()

    if fused:
        before = count_flcr(
            model
        )

        if before != 2:
            raise RuntimeError(
                f"expected 2 FLCR branches before fusion, got {before}"
            )

        fuse_flcr_inplace(
            model
        )

        after = count_flcr(
            model
        )

        if after != 0:
            raise RuntimeError(
                f"expected 0 FLCR branches after fusion, got {after}"
            )

        if adapter_state_keys(model):
            raise RuntimeError(
                "adapter state keys remain after fusion"
            )

        model.eval()

    return model


def count_flcr(model):
    return sum(
        isinstance(
            m,
            FLCRConv2d,
        )
        for m in model.modules()
    )


def adapter_state_keys(model):
    return [
        k
        for k in model.state_dict()
        if (
            ".repair_v." in k
            or ".repair_d." in k
            or ".repair_u." in k
        )
    ]


def prediction_arrays(sample):
    p = (
        sample.pred_instances
        .to("cpu")
    )

    b = (
        p.bboxes.numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    s = (
        p.scores.numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    l = (
        p.labels.numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    if len(s) == 0:
        return (
            np.zeros(
                (0, 4),
                dtype=np.float64,
            ),
            np.zeros(
                0,
                dtype=np.float64,
            ),
            np.zeros(
                0,
                dtype=np.int64,
            ),
        )

    order = np.lexsort(
        (
            b[:, 0],
            -s,
            l,
        )
    )

    return (
        b[order],
        s[order],
        l[order],
    )


def infer_one(
    model,
    image,
    precision,
):
    if precision == "fp32":
        ctx = contextlib.nullcontext()

    elif precision == "amp_fp16":
        ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    else:
        raise ValueError(
            precision
        )

    with torch.no_grad():
        with ctx:
            return inference_detector(
                model,
                image,
            )


def load_u00_helpers():
    gate_file = (
        ROOT
        / "scripts/tests/"
          "u00_flcr_final_gate.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "u00_final_gate_reuse",
            gate_file,
        )
    )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def fp32_detection_set_equivalence(
    unfused,
    fused,
    images,
):
    """
    Final-checkpoint full detector comparison.

    Reuses the U00 order-invariant detection-set matcher rather
    than comparing post-NMS arrays position-by-position.
    """

    u00 = load_u00_helpers()

    count_mismatch = 0
    label_mismatch_images = 0
    low_iou_match_images = 0

    min_matched_iou = 1.0
    max_box_diff = 0.0
    max_score_diff = 0.0

    old_cudnn = (
        torch.backends.cudnn.allow_tf32
    )
    old_matmul = (
        torch.backends.cuda.matmul.allow_tf32
    )

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        for idx, image in enumerate(
            images,
            start=1,
        ):
            with torch.no_grad():
                ou = inference_detector(
                    unfused,
                    image,
                )

                of = inference_detector(
                    fused,
                    image,
                )

            pu = ou.pred_instances
            pf = of.pred_instances

            if len(pu) != len(pf):
                count_mismatch += 1
            else:
                matched = (
                    u00.match_detection_sets(
                        pu,
                        pf,
                    )
                )

                min_matched_iou = min(
                    min_matched_iou,
                    float(
                        matched["min_iou"]
                    ),
                )

                max_box_diff = max(
                    max_box_diff,
                    float(
                        matched[
                            "max_box_diff"
                        ]
                    ),
                )

                max_score_diff = max(
                    max_score_diff,
                    float(
                        matched[
                            "max_score_diff"
                        ]
                    ),
                )

                if (
                    matched[
                        "label_mismatch_count"
                    ]
                    > 0
                ):
                    label_mismatch_images += 1

                if (
                    matched["min_iou"]
                    < 0.999
                ):
                    low_iou_match_images += 1

            if idx % 10 == 0:
                print(
                    "FP32_SET_EQUIV",
                    idx,
                    "/",
                    len(images),
                    flush=True,
                )

    finally:
        torch.backends.cudnn.allow_tf32 = (
            old_cudnn
        )

        torch.backends.cuda.matmul.allow_tf32 = (
            old_matmul
        )

    passed = bool(
        count_mismatch == 0
        and label_mismatch_images == 0
        and low_iou_match_images == 0
        and min_matched_iou >= 0.999
        and max_score_diff <= 1e-4
    )

    return {
        "comparison":
            "order_invariant_post_nms_detection_set",
        "images":
            len(images),
        "count_mismatch":
            count_mismatch,
        "label_mismatch_images":
            label_mismatch_images,
        "low_iou_match_images":
            low_iou_match_images,
        "min_matched_iou":
            float(min_matched_iou),
        "max_abs_bbox_diff_pixels":
            float(max_box_diff),
        "max_abs_score_diff":
            float(max_score_diff),
        "matched_iou_floor":
            0.999,
        "matched_score_guard":
            1e-4,
        "pass":
            passed,
    }


def actual_flcr_module_equivalence(
    model,
):
    """
    Apply the exact frozen U00-05 numerical fusion test
    to the actual trained K00-01 FLCR weights.

    Important:
    - U00-05 disables TF32 for the FP32 hard numeric gate.
    - FP32 < 1e-4 is the hard route requirement.
    - FP16 < 1e-3 is the recorded recommended target;
      U00 already classified a miss as nonblocking.
    """

    gate_file = (
        ROOT
        / "scripts/tests/"
          "u00_flcr_core_gate.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "u00_core_gate_reuse",
            gate_file,
        )
    )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    selected = [
        "bbox_head.cls_convs.0.1.conv",
        "bbox_head.reg_convs.0.1.conv",
    ]

    per_layer, fp32_max, fp16_max = (
        mod.actual_module_fusion_tests(
            model,
            selected,
            "cuda:0",
        )
    )

    fp32_threshold = 1e-4
    fp16_recommended = 1e-3

    return {
        "measurement_protocol":
            "exact reuse of frozen U00-05 "
            "actual_module_fusion_tests",
        "selected_layers":
            selected,
        "fp32": {
            "max_abs_error":
                float(fp32_max),
            "hard_threshold":
                fp32_threshold,
            "tf32_disabled":
                True,
            "hard_gate_pass":
                bool(
                    fp32_max
                    < fp32_threshold
                ),
        },
        "fp16": {
            "max_abs_error":
                float(fp16_max),
            "recommended_threshold":
                fp16_recommended,
            "target_met":
                bool(
                    fp16_max
                    < fp16_recommended
                ),
            "status":
                (
                    "TARGET_MET"
                    if fp16_max
                    < fp16_recommended
                    else
                    "TARGET_MISS_NONBLOCKING"
                ),
            "blocking":
                False,
        },
        "per_layer":
            per_layer,
    }


def raw_occlusion_map(
    data,
):
    result = {}
    missing = 0
    total = 0

    anns_by_img = defaultdict(
        list
    )

    for ann in data[
        "annotations"
    ]:
        anns_by_img[
            int(
                ann["image_id"]
            )
        ].append(ann)

    images = {
        int(im["id"]): im
        for im in data[
            "images"
        ]
    }

    for iid, anns in (
        anns_by_img.items()
    ):
        im = images[iid]

        stem = Path(
            im["file_name"]
        ).stem

        path = (
            RAW_ANN
            / f"{stem}.txt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        rows = np.loadtxt(
            path,
            delimiter=",",
            ndmin=2,
            dtype=np.float64,
        )

        lookup = defaultdict(list)

        for row in rows:
            if len(row) < 8:
                continue

            cat = int(
                round(
                    row[5]
                )
            )

            if cat < 1 or cat > 10:
                continue

            key = (
                round(
                    float(
                        row[0]
                    ),
                    3,
                ),
                round(
                    float(
                        row[1]
                    ),
                    3,
                ),
                round(
                    float(
                        row[2]
                    ),
                    3,
                ),
                round(
                    float(
                        row[3]
                    ),
                    3,
                ),
                cat,
            )

            lookup[key].append(
                int(
                    round(
                        row[7]
                    )
                )
            )

        for ann in anns:
            total += 1

            box = ann[
                "bbox"
            ]

            key = (
                round(
                    float(box[0]),
                    3,
                ),
                round(
                    float(box[1]),
                    3,
                ),
                round(
                    float(box[2]),
                    3,
                ),
                round(
                    float(box[3]),
                    3,
                ),
                int(
                    ann[
                        "category_id"
                    ]
                ),
            )

            candidates = lookup.get(
                key,
                [],
            )

            if not candidates:
                missing += 1
                continue

            occ = candidates.pop(0)

            result[
                int(
                    ann["id"]
                )
            ] = int(occ)

    return (
        result,
        {
            "total_coco_gt":
                total,
            "matched_occlusion":
                len(result),
            "missing":
                missing,
            "coverage":
                (
                    len(result)
                    / total
                    if total
                    else 0.0
                ),
        },
    )


def iou(
    box,
    gt,
):
    x1 = np.maximum(
        box[0],
        gt[:, 0],
    )

    y1 = np.maximum(
        box[1],
        gt[:, 1],
    )

    x2 = np.minimum(
        box[0] + box[2],
        gt[:, 0]
        + gt[:, 2],
    )

    y2 = np.minimum(
        box[1] + box[3],
        gt[:, 1]
        + gt[:, 3],
    )

    iw = np.maximum(
        0.0,
        x2 - x1,
    )

    ih = np.maximum(
        0.0,
        y2 - y1,
    )

    inter = iw * ih

    union = (
        box[2] * box[3]
        + gt[:, 2] * gt[:, 3]
        - inter
    )

    return np.divide(
        inter,
        union,
        out=np.zeros_like(
            inter,
            dtype=np.float64,
        ),
        where=union > 0,
    )


def occlusion_recall(
    data,
    occ_map,
    prediction_file,
):
    p = np.load(
        prediction_file
    )

    gt = defaultdict(list)

    for ann in data[
        "annotations"
    ]:
        aid = int(
            ann["id"]
        )

        if aid not in occ_map:
            continue

        key = (
            int(
                ann["image_id"]
            ),
            int(
                ann["category_id"]
            ),
        )

        gt[key].append({
            "bbox":
                np.asarray(
                    ann["bbox"],
                    dtype=np.float64,
                ),
            "occ":
                int(
                    occ_map[aid]
                ),
        })

    dt = defaultdict(list)

    for iid, cid, box, score in zip(
        p["image_id"],
        p["category_id"],
        p["bbox"],
        p["score"],
    ):
        dt[
            (
                int(iid),
                int(cid),
            )
        ].append(
            (
                float(score),
                np.asarray(
                    box,
                    dtype=np.float64,
                ),
            )
        )

    total = defaultdict(int)
    matched = defaultdict(int)

    for key, rows in gt.items():
        boxes = np.stack([
            x["bbox"]
            for x in rows
        ])

        occs = [
            x["occ"]
            for x in rows
        ]

        used = np.zeros(
            len(rows),
            dtype=bool,
        )

        for o in occs:
            total[o] += 1

        pred_rows = sorted(
            dt.get(
                key,
                [],
            ),
            key=lambda x:
                -x[0],
        )

        for score, box in pred_rows:
            v = iou(
                box,
                boxes,
            )

            v[used] = -1

            if len(v) == 0:
                continue

            j = int(
                np.argmax(v)
            )

            if v[j] >= 0.5:
                used[j] = True
                matched[
                    occs[j]
                ] += 1

    return {
        str(k): {
            "matched_gt":
                int(
                    matched[k]
                ),
            "total_gt":
                int(
                    total[k]
                ),
            "recall50":
                (
                    float(
                        matched[k]
                        / total[k]
                    )
                    if total[k]
                    else 0.0
                ),
        }
        for k in sorted(
            total
        )
    }


def x02_occlusion(
    out,
    data,
):
    occ_map, audit = (
        raw_occlusion_map(
            data
        )
    )

    if audit[
        "coverage"
    ] < 0.999:
        raise RuntimeError(
            f"occlusion mapping "
            f"coverage too low: "
            f"{audit}"
        )

    result = {
        "source":
            "original VisDrone "
            "test-dev annotation "
            "column 8",
        "matching":
            "class-aware greedy "
            "IoU>=0.5",
        "mapping_audit":
            audit,
        "models": {},
    }

    for mid in [
        "baseline",
        "equalstep_fullft",
        "kstar_seed0",
    ]:
        pred = (
            X01X02
            / mid
            / "clean_predictions.npz"
        )

        result[
            "models"
        ][mid] = (
            occlusion_recall(
                data,
                occ_map,
                pred,
            )
        )

    p = (
        out
        / "x02_visdrone_occlusion.json"
    )

    p.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return result


def export_onnx(
    fused,
    output_file,
):
    output_file = Path(output_file).resolve()

    output_file = Path(
        output_file
    ).resolve()

    gate_file = (
        ROOT
        / "scripts/tests/"
          "u00_flcr_final_gate.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "u00_final_gate_reuse",
            gate_file,
        )
    )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    ret = mod.onnx_smoke(
        fused,
        output_file,
    )

    if (
        not output_file.is_file()
        or output_file.stat().st_size
        == 0
    ):
        raise RuntimeError(
            "ONNX export missing"
        )

    return {
        "status": "PASS",
        "file":
            str(
                output_file.relative_to(
                    ROOT
                )
            ),
        "sha256":
            sha256(
                output_file
            ),
        "bytes":
            output_file.stat().st_size,
        "helper_return":
            (
                ret
                if isinstance(
                    ret,
                    (
                        str,
                        int,
                        float,
                        bool,
                        list,
                        dict,
                        type(None),
                    ),
                )
                else repr(ret)
            ),
    }


def benchmark_e2e_fp32(
    model,
    image,
    warmup=200,
    iterations=1000,
):
    """
    Full route-compliant latency:
    preprocessing + model + decode/NMS/postprocess.
    """

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        with torch.no_grad():
            inference_detector(
                model,
                image,
            )

    torch.cuda.synchronize()

    times_ms = []

    start_all = time.perf_counter()

    for _ in range(iterations):
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            inference_detector(
                model,
                image,
            )

        torch.cuda.synchronize()

        times_ms.append(
            (
                time.perf_counter()
                - t0
            )
            * 1000.0
        )

    total = (
        time.perf_counter()
        - start_all
    )

    arr = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    peak = (
        torch.cuda
        .max_memory_allocated()
        / (1024 ** 2)
    )

    return {
        "measurement_scope":
            "end_to_end_preprocess_model_nms_postprocess",
        "precision":
            "fp32",
        "batch_size":
            1,
        "warmup":
            warmup,
        "iterations":
            iterations,
        "cuda_synchronize_each_iteration":
            True,
        "p50_ms":
            float(
                np.percentile(
                    arr,
                    50,
                )
            ),
        "p95_ms":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),
        "mean_ms":
            float(
                np.mean(arr)
            ),
        "throughput_images_per_second":
            float(
                iterations
                / total
            ),
        "peak_allocated_vram_mb":
            float(peak),
    }


def benchmark_fp16_model_forward(
    model,
    warmup=200,
    iterations=1000,
):
    """
    FP16 diagnostic only.

    Current MMCV CUDA NMS requires Float, so this intentionally
    excludes decode/NMS/postprocess and is never used for the
    route latency safety gate.
    """

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device="cuda",
        dtype=torch.float32,
    )

    def one():
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                return model.bbox_head(
                    model.extract_feat(x)
                )

    for _ in range(warmup):
        one()

    torch.cuda.synchronize()

    times_ms = []

    start_all = time.perf_counter()

    for _ in range(iterations):
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        one()

        torch.cuda.synchronize()

        times_ms.append(
            (
                time.perf_counter()
                - t0
            )
            * 1000.0
        )

    total = (
        time.perf_counter()
        - start_all
    )

    arr = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    peak = (
        torch.cuda
        .max_memory_allocated()
        / (1024 ** 2)
    )

    return {
        "measurement_scope":
            "model_forward_only_no_decode_nms_postprocess",
        "precision":
            "amp_fp16",
        "batch_size":
            1,
        "warmup":
            warmup,
        "iterations":
            iterations,
        "cuda_synchronize_each_iteration":
            True,
        "nms_excluded_reason":
            "installed MMCV CUDA NMS requires Float tensors",
        "route_latency_safety_metric":
            False,
        "p50_ms":
            float(
                np.percentile(
                    arr,
                    50,
                )
            ),
        "p95_ms":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),
        "mean_ms":
            float(
                np.mean(arr)
            ),
        "throughput_images_per_second":
            float(
                iterations
                / total
            ),
        "peak_allocated_vram_mb":
            float(peak),
    }


def freeze_protocol(path):
    if sha256(
        BASELINE
    ) != EXPECTED[
        "baseline"
    ]:
        raise RuntimeError(
            "baseline SHA mismatch"
        )

    if sha256(
        KSTAR
    ) != EXPECTED[
        "kstar"
    ]:
        raise RuntimeError(
            "K* SHA mismatch"
        )

    data = load_json(
        ANN
    )

    images = sorted(
        data["images"],
        key=lambda x:
            int(x["id"]),
    )

    if len(images) != 1610:
        raise RuntimeError(
            "test-dev image count mismatch"
        )

    eq_ids = [
        int(x["id"])
        for x in images[:50]
    ]

    latency_id = int(
        images[0]["id"]
    )

    protocol = {
        "stage": "X00",
        "substage":
            "X02_occlusion_X03_X04",
        "status":
            "FROZEN_BEFORE_X03_X04_RESULTS",
        "remaining_training": 0,
        "x02_occlusion": {
            "source":
                "original VisDrone "
                "test-dev annotation "
                "column 8",
            "prediction_source":
                "frozen X01/X02 clean predictions",
            "matching":
                "class-aware greedy IoU>=0.5",
            "reinference": False,
        },
        "x03": {
            "model":
                "K00-01 frozen K*",
            "checkpoint_sha256":
                EXPECTED["kstar"],
            "expected_flcr_branches_before":
                2,
            "expected_flcr_branches_after":
                0,
            "strict_fusion_correctness_evidence":
                str(
                    U00_EVIDENCE.relative_to(
                        ROOT
                    )
                ),
            "strict_fusion_evidence_sha256":
                sha256(
                    U00_EVIDENCE
                ),
            "final_checkpoint_equivalence_images":
                eq_ids,
            "precisions":
                [
                    "fp32",
                    "amp_fp16",
                ],
            "onnx_required":
                True,
        },
        "x04": {
            "hardware":
                torch.cuda.get_device_name(
                    0
                ),
            "models": [
                "baseline",
                "kstar_unfused",
                "kstar_fused",
            ],
            "precisions": [
                "fp32_end_to_end",
                "amp_fp16_model_forward_diagnostic",
            ],
            "batch_size": 1,
            "input_pipeline":
                "frozen 640x640 config",
            "latency_image_id":
                latency_id,
            "warmup": 200,
            "iterations": 1000,
            "cuda_synchronize":
                True,
            "report": [
                "P50",
                "P95",
                "throughput",
                "peak_vram",
            ],
            "safety_constraint":
                "fused K* P50 relative "
                "to baseline should not "
                "degrade >2%",
        },
        "test_adaptive_use":
            False,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "X03_X04_PROTOCOL_FREEZE=PASS"
    )

    print(
        "GPU=",
        protocol[
            "x04"
        ][
            "hardware"
        ],
    )

    print(
        "EQUIVALENCE_IMAGES=50"
    )

    print(
        "LATENCY_IMAGE_ID=",
        latency_id,
    )


def smoke():
    u = build_kstar(
        fused=False
    )

    if count_flcr(u) != 2:
        raise RuntimeError(
            "unfused FLCR count mismatch"
        )

    f = copy.deepcopy(u)

    before = count_flcr(
        f
    )

    fuse_flcr_inplace(
        f
    )

    after = count_flcr(
        f
    )

    replaced = before - after

    if before != 2 or after != 0 or replaced != 2:
        raise RuntimeError(
            f"fusion count mismatch: before={before} after={after}"
        )

    if count_flcr(f) != 0:
        raise RuntimeError(
            "FLCR remains after fusion"
        )

    if adapter_state_keys(f):
        raise RuntimeError(
            "adapter state keys remain"
        )

    print(
        "MODEL_SMOKE=PASS"
    )

    print(
        "FLCR_BEFORE=2"
    )

    print(
        "FLCR_AFTER=0"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--freeze-protocol",
    )

    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    ap.add_argument(
        "--output",
    )

    args = ap.parse_args()

    register_all_modules(
        init_default_scope=True
    )

    if args.freeze_protocol:
        freeze_protocol(
            Path(
                args.freeze_protocol
            )
        )
        return

    if args.smoke:
        smoke()
        return

    if not args.output:
        raise RuntimeError(
            "--output required"
        )

    out = Path(
        args.output
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    data = load_json(
        ANN
    )

    images = sorted(
        data["images"],
        key=lambda x:
            int(x["id"]),
    )

    by_id = {
        int(x["id"]): x
        for x in images
    }

    protocol = load_json(
        ROOT
        / "analysis/x00/"
          "x03_x04_protocol_freeze.json"
    )

    print(
        "X02_OCCLUSION_START",
        flush=True,
    )

    occ = x02_occlusion(
        out,
        data,
    )

    print(
        "X02_OCCLUSION_COMPLETE",
        occ[
            "mapping_audit"
        ],
        flush=True,
    )

    print(
        "X03_BUILD_START",
        flush=True,
    )

    unfused = build_kstar(
        fused=False
    )

    if count_flcr(
        unfused
    ) != 2:
        raise RuntimeError(
            "formal unfused count mismatch"
        )

    fused = copy.deepcopy(
        unfused
    )

    before = count_flcr(
        fused
    )

    fuse_flcr_inplace(
        fused
    )

    after = count_flcr(
        fused
    )

    replaced = before - after

    if before != 2 or after != 0 or replaced != 2:
        raise RuntimeError(
            f"formal fusion count mismatch: before={before} after={after}"
        )

    fused.eval()

    keys_after = (
        adapter_state_keys(
            fused
        )
    )

    if after != 0:
        raise RuntimeError(
            "formal fused branches remain"
        )

    if keys_after:
        raise RuntimeError(
            "formal adapter keys remain"
        )

    eq_images = []

    for iid in protocol[
        "x03"
    ][
        "final_checkpoint_equivalence_images"
    ]:
        x = cv2.imread(
            str(
                image_path(
                    by_id[
                        int(iid)
                    ]
                )
            ),
            cv2.IMREAD_COLOR,
        )

        if x is None:
            raise RuntimeError(
                f"image load failed {iid}"
            )

        eq_images.append(x)

    module_eq = (
        actual_flcr_module_equivalence(
            unfused
        )
    )

    print(
        "MODULE_EQUIV_FP32",
        module_eq["fp32"],
        flush=True,
    )

    print(
        "MODULE_EQUIV_FP16",
        module_eq["fp16"],
        flush=True,
    )

    if not module_eq[
        "fp32"
    ][
        "hard_gate_pass"
    ]:
        raise RuntimeError(
            "actual K* FLCR FP32 U00-05 "
            "hard fusion gate failed"
        )

    detection_eq = (
        fp32_detection_set_equivalence(
            unfused,
            fused,
            eq_images,
        )
    )

    print(
        "FP32_DETECTION_SET_EQUIV",
        detection_eq,
        flush=True,
    )

    if not detection_eq["pass"]:
        raise RuntimeError(
            "final K* FP32 detection-set equivalence failed"
        )

    eq = {
        "actual_module_equivalence":
            module_eq,
        "fp32_detection_set_equivalence":
            detection_eq,
        "fp16_post_nms":
            {
                "status":
                    "NOT_RUN",
                "reason":
                    "installed MMCV CUDA NMS requires Float tensors",
            },
    }

    onnx_file = (
        out
        / "kstar_fused.onnx"
    )

    onnx = export_onnx(
        fused,
        onnx_file,
    )

    x03 = {
        "checkpoint":
            str(
                KSTAR.relative_to(
                    ROOT
                )
            ),
        "checkpoint_sha256":
            sha256(KSTAR),
        "flcr_branches_before":
            2,
        "fused_modules":
            replaced,
        "flcr_branches_after":
            after,
        "adapter_state_keys_after":
            keys_after,
        "strict_U00_evidence": {
            "file":
                str(
                    U00_EVIDENCE.relative_to(
                        ROOT
                    )
                ),
            "sha256":
                sha256(
                    U00_EVIDENCE
                ),
        },
        "final_checkpoint_equivalence":
            eq,
        "onnx":
            onnx,
    }

    (
        out
        / "x03_fusion_onnx.json"
    ).write_text(
        json.dumps(
            x03,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "X03_COMPLETE",
        flush=True,
    )

    del unfused
    del fused

    torch.cuda.empty_cache()

    latency_im = by_id[
        int(
            protocol[
                "x04"
            ][
                "latency_image_id"
            ]
        )
    ]

    image = cv2.imread(
        str(
            image_path(
                latency_im
            )
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "latency image failed"
        )

    latency = {}

    for kind in [
        "baseline",
        "kstar_unfused",
        "kstar_fused",
    ]:
        print(
            "LATENCY_MODEL_START",
            kind,
            flush=True,
        )

        if kind == "baseline":
            model = (
                build_baseline()
            )

        elif kind == (
            "kstar_unfused"
        ):
            model = build_kstar(
                fused=False
            )

        else:
            model = build_kstar(
                fused=True
            )

        latency[kind] = {}

        print(
            "LATENCY_START",
            kind,
            "fp32_end_to_end",
            flush=True,
        )

        latency[kind][
            "fp32_end_to_end"
        ] = benchmark_e2e_fp32(
            model,
            image,
            warmup=200,
            iterations=1000,
        )

        print(
            "LATENCY_RESULT",
            kind,
            "fp32_end_to_end",
            latency[kind][
                "fp32_end_to_end"
            ],
            flush=True,
        )

        print(
            "LATENCY_START",
            kind,
            "fp16_model_forward",
            flush=True,
        )

        latency[kind][
            "fp16_model_forward"
        ] = (
            benchmark_fp16_model_forward(
                model,
                warmup=200,
                iterations=1000,
            )
        )

        print(
            "LATENCY_RESULT",
            kind,
            "fp16_model_forward",
            latency[kind][
                "fp16_model_forward"
            ],
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

    b = latency[
        "baseline"
    ][
        "fp32_end_to_end"
    ][
        "p50_ms"
    ]

    f = latency[
        "kstar_fused"
    ][
        "fp32_end_to_end"
    ][
        "p50_ms"
    ]

    u = latency[
        "kstar_unfused"
    ][
        "fp32_end_to_end"
    ][
        "p50_ms"
    ]

    safety = {
        "fp32_end_to_end": {
            "fused_vs_baseline_p50_pct":
                (
                    (f / b - 1.0)
                    * 100.0
                ),
            "unfused_vs_baseline_p50_pct":
                (
                    (u / b - 1.0)
                    * 100.0
                ),
            "fused_vs_unfused_p50_pct":
                (
                    (f / u - 1.0)
                    * 100.0
                ),
            "route_latency_safety_pass":
                bool(
                    (
                        f / b - 1.0
                    )
                    <= 0.02
                ),
        },
        "fp16_model_forward": {
            "route_latency_safety_metric":
                False,
            "reason":
                "post-NMS FP16 unavailable because installed MMCV NMS requires Float",
        },
    }

    x04 = {
        "hardware":
            torch.cuda.get_device_name(
                0
            ),
        "batch_size": 1,
        "input": "640x640 pipeline",
        "latency_image_id":
            int(
                protocol[
                    "x04"
                ][
                    "latency_image_id"
                ]
            ),
        "results":
            latency,
        "comparisons":
            safety,
    }

    (
        out
        / "x04_latency.json"
    ).write_text(
        json.dumps(
            x04,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    final = {
        "stage": "X00",
        "x02_occlusion":
            occ,
        "x03":
            x03,
        "x04":
            x04,
        "remaining_training": 0,
    }

    (
        out
        / "x03_x04_final.json"
    ).write_text(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "X03_X04=COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
