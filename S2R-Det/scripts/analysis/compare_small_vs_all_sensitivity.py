#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.apis import init_detector
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import register_all_modules


EPS = 1e-12
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 0
STABLE_FREQ = 0.60
COCO_SMALL_AREA = 32 * 32


def load_fcsl_module():
    path = Path(__file__).with_name(
        "run_fcsl_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "s2r_fcsl_reference",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_instance_loss(
    head,
    flatten_cls,
    flatten_bbox,
    gt_box,
    gt_label,
):
    with torch.no_grad():
        ix1 = torch.maximum(
            flatten_bbox[:, 0],
            gt_box[0],
        )
        iy1 = torch.maximum(
            flatten_bbox[:, 1],
            gt_box[1],
        )
        ix2 = torch.minimum(
            flatten_bbox[:, 2],
            gt_box[2],
        )
        iy2 = torch.minimum(
            flatten_bbox[:, 3],
            gt_box[3],
        )

        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih

        pa = (
            (flatten_bbox[:, 2] - flatten_bbox[:, 0])
            .clamp(min=0)
            * (flatten_bbox[:, 3] - flatten_bbox[:, 1])
            .clamp(min=0)
        )
        ga = (
            (gt_box[2] - gt_box[0]).clamp(min=0)
            * (gt_box[3] - gt_box[1]).clamp(min=0)
        )

        iou = inter / (
            pa + ga - inter
        ).clamp(min=1e-12)

        match_idx = int(
            torch.argmax(iou).item()
        )
        match_iou = float(
            iou[match_idx].item()
        )

    pred_logits = flatten_cls[
        match_idx:match_idx + 1
    ]
    pred_box = flatten_bbox[
        match_idx:match_idx + 1
    ]

    label = torch.tensor(
        [int(gt_label)],
        dtype=torch.long,
        device=pred_logits.device,
    )
    quality = torch.ones(
        1,
        dtype=pred_logits.dtype,
        device=pred_logits.device,
    )
    cls_weight = torch.ones(
        1,
        dtype=pred_logits.dtype,
        device=pred_logits.device,
    )

    loss_cls = head.loss_cls(
        pred_logits,
        (label, quality),
        cls_weight,
        avg_factor=1.0,
    )

    bbox_weight = torch.ones(
        1,
        dtype=pred_box.dtype,
        device=pred_box.device,
    )

    loss_bbox = head.loss_bbox(
        pred_box,
        gt_box[None],
        weight=bbox_weight,
        avg_factor=1.0,
    )

    return (
        loss_cls + loss_bbox,
        match_idx,
        match_iou,
    )


def bootstrap_small_vs_all(
    records,
    layer_names,
    analysis_meta,
):
    g = np.asarray(
        [r["g"] for r in records],
        dtype=np.float64,
    )
    logg = np.log(g + EPS)

    small = np.asarray([
        bool(r["is_coco_small"])
        for r in records
    ])

    if small.sum() == 0:
        raise RuntimeError(
            "small-vs-all contains no COCO-small GT"
        )

    observed = (
        np.median(logg[small], axis=0)
        - np.median(logg, axis=0)
    )

    rec_by_image = defaultdict(list)
    for i, r in enumerate(records):
        rec_by_image[r["file_name"]].append(i)

    strata = defaultdict(list)
    for fn, meta in analysis_meta.items():
        strata[
            (meta["cohort"], meta["density"])
        ].append(fn)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    q_boot = np.zeros(
        (BOOTSTRAPS, len(layer_names)),
        dtype=np.float64,
    )
    top1 = np.zeros(
        len(layer_names),
        dtype=np.int64,
    )
    top2 = np.zeros(
        len(layer_names),
        dtype=np.int64,
    )

    for b in range(BOOTSTRAPS):
        indices = []

        for key, names in sorted(strata.items()):
            sampled = rng.choice(
                np.asarray(names, dtype=object),
                size=len(names),
                replace=True,
            )

            for fn in sampled:
                indices.extend(
                    rec_by_image[str(fn)]
                )

        idx = np.asarray(
            indices,
            dtype=np.int64,
        )
        sm = small[idx]

        if sm.sum() == 0:
            raise RuntimeError(
                "bootstrap replicate has no small GT"
            )

        qb = (
            np.median(
                logg[idx][sm],
                axis=0,
            )
            - np.median(
                logg[idx],
                axis=0,
            )
        )

        q_boot[b] = qb

        order = np.argsort(
            -qb,
            kind="stable",
        )
        top1[order[0]] += 1
        top2[order[:2]] += 1

    return {
        "observed": observed,
        "bootstrap_median":
            np.median(q_boot, axis=0),
        "ci_low":
            np.quantile(q_boot, 0.025, axis=0),
        "ci_high":
            np.quantile(q_boot, 0.975, axis=0),
        "top1_freq":
            top1 / BOOTSTRAPS,
        "top2_freq":
            top2 / BOOTSTRAPS,
        "q_boot": q_boot,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eligible", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
    )
    p.add_argument(
        "--skip-bootstrap",
        action="store_true",
    )

    args = p.parse_args()

    register_all_modules(
        init_default_scope=True
    )

    fcsl = load_fcsl_module()

    eligible_doc = json.loads(
        Path(args.eligible).read_text(
            encoding="utf-8"
        )
    )
    eligible_layers = (
        eligible_doc["eligible_layers"]
    )
    layer_names = [
        x["name"]
        for x in eligible_layers
    ]
    by_name = {
        x["name"]: x
        for x in eligible_layers
    }

    analysis_rows = list(
        csv.DictReader(
            Path(args.analysis).open(
                newline="",
                encoding="utf-8",
            )
        )
    )

    if len(analysis_rows) != 512:
        raise RuntimeError(
            "A00-01 requires frozen 512-image S00 set"
        )

    if args.max_images > 0:
        analysis_rows = (
            analysis_rows[:args.max_images]
        )

    analysis_meta = {
        r["file_name"]: {
            "cohort": r["cohort"],
            "density": r["density"],
        }
        for r in analysis_rows
    }

    cfg = Config.fromfile(
        args.config
    )

    dataset = fcsl.make_dataset(
        cfg,
        args.data_root,
        args.train_json,
    )

    name_to_idx = {}
    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        name_to_idx[
            Path(info["img_path"]).name
        ] = i

    model = init_detector(
        args.config,
        args.checkpoint,
        device="cuda:0",
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    modules = dict(
        model.named_modules()
    )

    missing = [
        x
        for x in layer_names
        if x not in modules
    ]
    if missing:
        raise RuntimeError(
            "eligible layers missing: "
            + repr(missing)
        )

    activation_map = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            activation_map.setdefault(
                name, []
            ).append(output)
        return hook

    for name in layer_names:
        hooks.append(
            modules[name]
            .register_forward_hook(
                make_hook(name)
            )
        )

    records = []
    nonfinite_loss = 0
    nonfinite_g = 0
    zero_assignment = 0

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    try:
        for img_no, row in enumerate(
            analysis_rows, 1
        ):
            fn = row["file_name"]

            if fn not in name_to_idx:
                raise RuntimeError(
                    "dataset index missing: " + fn
                )

            item = dataset[
                name_to_idx[fn]
            ]
            batch = pseudo_collate([item])

            processed = (
                model.data_preprocessor(
                    batch,
                    training=False,
                )
            )

            inputs = processed["inputs"]
            inputs.requires_grad_(True)
            sample = (
                processed["data_samples"][0]
            )

            activation_map.clear()

            feats = model.extract_feat(inputs)
            cls_scores, bbox_preds = (
                model.bbox_head(feats)
            )

            dense_cls, dense_bbox = (
                fcsl.flatten_dense_predictions(
                    model.bbox_head,
                    cls_scores,
                    bbox_preds,
                    sample,
                )
            )

            gt = sample.gt_instances
            gt_boxes = get_box_tensor(
                gt.bboxes
            )
            gt_labels = gt.labels

            sf = sample.metainfo.get(
                "scale_factor",
                (1.0, 1.0),
            )

            sx = float(sf[0])
            sy = float(sf[1])

            if sx <= 0 or sy <= 0:
                raise RuntimeError(
                    "invalid scale_factor"
                )

            for gj in range(
                len(gt_labels)
            ):
                box = gt_boxes[gj]
                label = int(
                    gt_labels[gj].item()
                )

                scaled_area = float(
                    (
                        (box[2] - box[0])
                        .clamp(min=0)
                        * (box[3] - box[1])
                        .clamp(min=0)
                    ).detach().item()
                )

                original_area = (
                    scaled_area / (sx * sy)
                )

                loss, match_idx, match_iou = (
                    dense_instance_loss(
                        model.bbox_head,
                        dense_cls,
                        dense_bbox,
                        box,
                        label,
                    )
                )

                if loss is None:
                    zero_assignment += 1
                    continue

                loss_value = float(
                    loss.detach().item()
                )

                if not math.isfinite(
                    loss_value
                ):
                    nonfinite_loss += 1
                    continue

                g = fcsl.sensitivity_for_loss(
                    loss,
                    activation_map,
                    layer_names,
                )

                if not np.all(
                    np.isfinite(g)
                ):
                    nonfinite_g += 1
                    continue

                records.append({
                    "file_name": fn,
                    "cohort": row["cohort"],
                    "density": row["density"],
                    "gt_index": gj,
                    "category_id": label,
                    "original_area":
                        original_area,
                    "is_coco_small":
                        original_area
                        < COCO_SMALL_AREA,
                    "dense_match_index":
                        int(match_idx),
                    "dense_match_iou":
                        float(match_iou),
                    "instance_loss":
                        loss_value,
                    "g": g,
                })

            del (
                feats,
                cls_scores,
                bbox_preds,
                dense_cls,
                dense_bbox,
                inputs,
            )
            torch.cuda.empty_cache()

            if (
                img_no % 10 == 0
                or img_no
                == len(analysis_rows)
            ):
                print(
                    "PROGRESS="
                    f"{img_no}/"
                    f"{len(analysis_rows)} "
                    f"instances={len(records)} "
                    f"elapsed_min="
                    f"{(time.time()-t0)/60:.2f}",
                    flush=True,
                )

    finally:
        for h in hooks:
            h.remove()

    valid = (
        zero_assignment == 0
        and nonfinite_loss == 0
        and nonfinite_g == 0
        and len(records) > 0
    )

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_rows = []
    for r in records:
        x = {
            k: v
            for k, v in r.items()
            if k != "g"
        }
        for i, value in enumerate(
            r["g"]
        ):
            x[
                f"g_{i:03d}"
            ] = float(value)
        parquet_rows.append(x)

    pq.write_table(
        pa.Table.from_pylist(
            parquet_rows
        ),
        out
        / "instance_sensitivity.parquet",
        compression="zstd",
    )

    result = {
        "stage": "A00",
        "family": "A00-01",
        "contrast":
            "small_vs_all",
        "definition":
            "median log(g) over COCO-small GT "
            "minus median log(g) over all valid GT; "
            "small is original-area < 32^2; "
            "all includes small",
        "analysis_set":
            "same frozen 512 S00 images",
        "processed_images":
            len(analysis_rows),
        "computed_instances":
            len(records),
        "small_instances":
            sum(
                r["is_coco_small"]
                for r in records
            ),
        "zero_assignment":
            zero_assignment,
        "nonfinite_loss":
            nonfinite_loss,
        "nonfinite_sensitivity":
            nonfinite_g,
        "engineering_gate":
            "PASS" if valid else "HOLD",
        "epsilon": EPS,
        "wall_time_seconds":
            time.time() - t0,
        "peak_vram_mb":
            int(
                torch.cuda
                .max_memory_allocated()
                / 1024 / 1024
            ),
    }

    if args.skip_bootstrap:
        (
            out / "run_summary.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            "A00_SMALL_VS_ALL_GATE="
            + result["engineering_gate"]
        )
        return

    if len(analysis_rows) != 512:
        raise RuntimeError(
            "bootstrap only valid on full 512 images"
        )

    if not valid:
        raise RuntimeError(
            "engineering gate failed"
        )

    boot = bootstrap_small_vs_all(
        records,
        layer_names,
        analysis_meta,
    )

    order = np.argsort(
        -boot["bootstrap_median"],
        kind="stable",
    )

    stable_top1 = [
        i for i in range(
            len(layer_names)
        )
        if boot["top1_freq"][i]
        >= STABLE_FREQ
    ]
    stable_top2 = [
        i for i in range(
            len(layer_names)
        )
        if boot["top2_freq"][i]
        >= STABLE_FREQ
    ]

    stable_top1.sort(
        key=lambda i: (
            -boot["bootstrap_median"][i],
            layer_names[i],
        )
    )
    stable_top2.sort(
        key=lambda i: (
            -boot["bootstrap_median"][i],
            layer_names[i],
        )
    )

    if len(stable_top2) >= 2:
        selected_idx = (
            stable_top2[:2]
        )
        selection_status = (
            "STABLE_TOP2"
        )
    elif stable_top1:
        selected_idx = [
            stable_top1[0]
        ]
        selection_status = (
            "STABLE_TOP1_ONLY"
        )
    else:
        selected_idx = list(
            order[:2]
        )
        selection_status = (
            "UNSTABLE_RAW_TOP2_CONTROL"
        )

    layer_rows = []
    for rank_no, i in enumerate(
        order, 1
    ):
        layer_rows.append({
            "rank": rank_no,
            "layer": layer_names[i],
            "q_observed":
                float(
                    boot["observed"][i]
                ),
            "q_bootstrap_median":
                float(
                    boot[
                        "bootstrap_median"
                    ][i]
                ),
            "q_ci_low":
                float(
                    boot["ci_low"][i]
                ),
            "q_ci_high":
                float(
                    boot["ci_high"][i]
                ),
            "top1_frequency":
                float(
                    boot["top1_freq"][i]
                ),
            "top2_frequency":
                float(
                    boot["top2_freq"][i]
                ),
        })

    with (
        out / "layer_stability.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                layer_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(layer_rows)

    np.savez_compressed(
        out / "bootstrap_q.npz",
        q=boot["q_boot"],
        layers=np.asarray(
            layer_names,
            dtype=object,
        ),
    )

    selected_layers = [
        layer_names[i]
        for i in selected_idx
    ]

    selected_params = int(
        sum(
            by_name[n][
                "flcr_rank8_parameters"
            ]
            for n in selected_layers
        )
    )

    selection = {
        "stage": "A00",
        "family": "A00-01",
        "selection_rule":
            "stable Top-2; if unavailable stable "
            "Top-1; otherwise raw median-score "
            "Top-2 labeled unstable",
        "selection_status":
            selection_status,
        "selected_layers":
            selected_layers,
        "rank": 8,
        "trainable_parameters":
            selected_params,
        "stable_frequency_threshold":
            STABLE_FREQ,
        "test_dev_used": False,
    }

    (
        out / "selection.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result.update({
        "bootstrap": {
            "replicates":
                BOOTSTRAPS,
            "seed":
                BOOTSTRAP_SEED,
            "strata":
                "original S00 cohort x density; "
                "image-level resampling",
            "stable_frequency_threshold":
                STABLE_FREQ,
        },
        "selection_status":
            selection_status,
        "selected_layers":
            selected_layers,
        "selected_rank":
            8,
        "selected_trainable_parameters":
            selected_params,
    })

    (
        out / "run_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "A00_SMALL_VS_ALL_GATE=PASS"
    )
    print(
        "A00_SMALL_VS_ALL_SELECTION="
        + selection_status
    )
    print(
        "A00_SMALL_VS_ALL_LAYERS="
        + "|".join(selected_layers)
    )


if __name__ == "__main__":
    main()
