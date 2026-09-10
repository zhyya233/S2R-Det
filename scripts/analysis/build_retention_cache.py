#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from mmdet.apis import (
    inference_detector,
    init_detector,
)

from s2r_det.training.prediction_retention import (
    flatten_dense_predictions,
    greedy_f5_match_state,
    reconstruct_ppre_dense_map,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_retention_names(path: Path):
    names = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        names.append(
            Path(line).name
        )

    return names


def load_f5_gt(parquet: Path):
    tab = pq.read_table(
        parquet,
        columns=[
            "file_name",
            "category_id",
            "gt_x1",
            "gt_y1",
            "gt_x2",
            "gt_y2",
            "failure_type",
        ],
    ).to_pydict()

    out = defaultdict(list)

    for (
        name,
        category,
        x1,
        y1,
        x2,
        y2,
        failure_type,
    ) in zip(
        tab["file_name"],
        tab["category_id"],
        tab["gt_x1"],
        tab["gt_y1"],
        tab["gt_x2"],
        tab["gt_y2"],
        tab["failure_type"],
    ):
        if failure_type != "F5":
            continue

        out[
            Path(name).name
        ].append(
            (
                int(category) - 1,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
        )

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument(
        "--train-root",
        required=True,
    )
    ap.add_argument(
        "--retention-list",
        required=True,
    )
    ap.add_argument(
        "--failure-parquet",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
    ap.add_argument(
        "--manifest",
        required=True,
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=100,
    )
    ap.add_argument(
        "--expected-checkpoint-sha",
        required=True,
    )
    ap.add_argument(
        "--expected-config-sha",
        required=True,
    )

    args = ap.parse_args()

    config = Path(args.config)
    checkpoint = Path(
        args.checkpoint
    )
    train_root = Path(
        args.train_root
    )
    retention_list = Path(
        args.retention_list
    )
    failure_parquet = Path(
        args.failure_parquet
    )
    output = Path(args.output)
    manifest_path = Path(
        args.manifest
    )

    if sha256(checkpoint) != args.expected_checkpoint_sha:
        raise RuntimeError(
            "checkpoint SHA mismatch"
        )

    if sha256(config) != args.expected_config_sha:
        raise RuntimeError(
            "config SHA mismatch"
        )

    names = load_retention_names(
        retention_list
    )

    if len(names) != 2275:
        raise RuntimeError(
            f"expected 2275 retention images, got {len(names)}"
        )

    if len(set(names)) != len(names):
        raise RuntimeError(
            "duplicate retention filenames"
        )

    f5_gt = load_f5_gt(
        failure_parquet
    )

    missing_f5 = [
        name
        for name in names
        if not f5_gt.get(name)
    ]

    if missing_f5:
        raise RuntimeError(
            "retention images without F5 target: "
            + repr(missing_f5[:10])
        )

    model = init_detector(
        str(config),
        str(checkpoint),
        device="cuda:0",
    )

    model.eval()

    captures = {}

    def hook(module, inputs, output_):
        captures["head"] = output_

    handle = (
        model.bbox_head
        .register_forward_hook(hook)
    )

    entries = {}

    max_ppre_score_diff = 0.0
    ppre_label_mismatch = 0
    ppre_count_mismatch_images = 0

    min_ppre_count = None
    max_ppre_count = 0

    total_gt_matched_cached = 0
    dense_duplicate_candidates = 0

    start = time.time()

    try:
        for i, name in enumerate(names):
            path = (
                train_root
                / "images"
                / name
            )

            if not path.is_file():
                raise FileNotFoundError(
                    str(path)
                )

            captures.clear()

            with torch.no_grad():
                result = inference_detector(
                    model,
                    str(path),
                )

            if isinstance(result, list):
                if len(result) != 1:
                    raise RuntimeError(
                        "unexpected inference result list size"
                    )

                result = result[0]

            if "head" not in captures:
                raise RuntimeError(
                    "bbox_head output capture failed"
                )

            raw = captures["head"]

            cls_scores = raw[0]
            bbox_preds = raw[1]

            metas = [
                dict(result.metainfo)
            ]

            with torch.no_grad():
                pre = (
                    model.bbox_head
                    .predict_by_feat(
                        *raw,
                        batch_img_metas=metas,
                        cfg=model.test_cfg,
                        rescale=True,
                        with_nms=False,
                    )[0]
                )

                flat_cls, flat_bbox = (
                    flatten_dense_predictions(
                        model.bbox_head,
                        cls_scores,
                        bbox_preds,
                        result,
                    )
                )

                (
                    reconstructed_scores,
                    reconstructed_labels,
                    reconstructed_dense,
                ) = reconstruct_ppre_dense_map(
                    model.bbox_head,
                    cls_scores,
                    bbox_preds,
                    dict(result.metainfo),
                    model.test_cfg,
                    rescale=True,
                )

            npre = len(pre)

            min_ppre_count = (
                npre
                if min_ppre_count is None
                else min(
                    min_ppre_count,
                    npre,
                )
            )

            max_ppre_count = max(
                max_ppre_count,
                npre,
            )

            if (
                reconstructed_scores.shape[0]
                != npre
            ):
                ppre_count_mismatch_images += 1

                raise RuntimeError(
                    f"{name}: reconstructed Ppre count "
                    f"{reconstructed_scores.shape[0]} != {npre}"
                )

            lm = int(
                (
                    reconstructed_labels
                    != pre.labels
                ).sum().item()
            )

            ppre_label_mismatch += lm

            if lm:
                raise RuntimeError(
                    f"{name}: Ppre label-order mismatch={lm}"
                )

            if npre:
                sd = float(
                    (
                        reconstructed_scores
                        - pre.scores
                    )
                    .abs()
                    .max()
                    .item()
                )

                max_ppre_score_diff = max(
                    max_ppre_score_diff,
                    sd,
                )

                if sd > 1e-7:
                    raise RuntimeError(
                        f"{name}: Ppre score reconstruction diff={sd}"
                    )

            k = min(
                int(args.topk),
                npre,
            )

            if k != args.topk:
                raise RuntimeError(
                    f"{name}: only {npre} Ppre candidates"
                )

            top_pre_idx = torch.argsort(
                pre.scores,
                descending=True,
                stable=True,
            )[:k]

            dense_idx = (
                reconstructed_dense[
                    top_pre_idx
                ]
            )

            dense_duplicate_candidates += (
                k
                - int(
                    torch.unique(
                        dense_idx
                    ).numel()
                )
            )

            teacher_logits = (
                flat_cls[
                    dense_idx
                ]
            )

            teacher_boxes_input = (
                flat_bbox[
                    dense_idx
                ]
            )

            teacher_boxes_original = (
                pre.bboxes[
                    top_pre_idx
                ]
            )

            teacher_labels = (
                pre.labels[
                    top_pre_idx
                ]
            )

            teacher_scores = (
                pre.scores[
                    top_pre_idx
                ]
            )

            rows = f5_gt[name]

            gt_labels = torch.tensor(
                [
                    r[0]
                    for r in rows
                ],
                device=teacher_boxes_original.device,
                dtype=torch.long,
            )

            gt_boxes = torch.tensor(
                [
                    r[1:]
                    for r in rows
                ],
                device=teacher_boxes_original.device,
                dtype=teacher_boxes_original.dtype,
            )

            match_state = (
                greedy_f5_match_state(
                    teacher_boxes_original,
                    teacher_labels,
                    gt_boxes,
                    gt_labels,
                    iou_thr=0.5,
                )
            )

            total_gt_matched_cached += int(
                match_state.sum().item()
            )

            entries[name] = {
                "pre_idx":
                    top_pre_idx
                    .to(
                        dtype=torch.int32,
                        device="cpu",
                    ),

                "dense_idx":
                    dense_idx
                    .to(
                        dtype=torch.int32,
                        device="cpu",
                    ),

                "teacher_label":
                    teacher_labels
                    .to(
                        dtype=torch.int16,
                        device="cpu",
                    ),

                "teacher_score":
                    teacher_scores
                    .to(
                        dtype=torch.float16,
                        device="cpu",
                    ),

                "logits":
                    teacher_logits
                    .to(
                        dtype=torch.float16,
                        device="cpu",
                    ),

                "boxes_input":
                    teacher_boxes_input
                    .to(
                        dtype=torch.float16,
                        device="cpu",
                    ),

                "boxes_original":
                    teacher_boxes_original
                    .to(
                        dtype=torch.float16,
                        device="cpu",
                    ),

                "gt_match":
                    match_state
                    .to(
                        dtype=torch.uint8,
                        device="cpu",
                    ),
            }

            if (
                (i + 1) % 250 == 0
                or i + 1 == len(names)
            ):
                print(
                    f"CACHE_PROGRESS={i+1}/{len(names)}",
                    flush=True,
                )

    finally:
        handle.remove()

    payload = {
        "meta": {
            "format_version":
                "p00-retention-v1",

            "checkpoint_sha256":
                args.expected_checkpoint_sha,

            "config_sha256":
                args.expected_config_sha,

            "retention_list_sha256":
                sha256(retention_list),

            "failure_parquet_sha256":
                sha256(failure_parquet),

            "retention_images":
                len(names),

            "topk":
                int(args.topk),

            "candidate_semantics":
                (
                    "pre_idx indexes frozen MMDetection "
                    "predict_by_feat(with_nms=False) Ppre; "
                    "dense_idx is its source dense RTMDet location"
                ),

            "topk_rule":
                (
                    "global descending Ppre score; "
                    "stable candidate-order tie break"
                ),

            "logit_semantics":
                "raw RTMDet sigmoid-head class logits",

            "boxes_input_semantics":
                (
                    "decoded dense boxes in model-input coordinates"
                ),

            "boxes_original_semantics":
                (
                    "corresponding rescale=True frozen Ppre boxes"
                ),

            "gt_match_semantics":
                (
                    "one-to-one, class-consistent, "
                    "IoU-priority match to F5 GT at IoU>=0.5"
                ),

            "float_cache_dtype":
                "float16",

            "ppre_score_reconstruction_tolerance":
                1e-7,
        },

        "entries": entries,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        output,
    )

    cache_sha = sha256(
        output
    )

    manifest = {
        **payload["meta"],

        "cache_path":
            str(output),

        "cache_sha256":
            cache_sha,

        "cache_size_bytes":
            output.stat().st_size,

        "ppre_reconstruction": {
            "count_mismatch_images":
                ppre_count_mismatch_images,

            "label_mismatch_total":
                ppre_label_mismatch,

            "max_score_abs_diff":
                max_ppre_score_diff,
        },

        "ppre_count_range": {
            "min":
                int(min_ppre_count),

            "max":
                int(max_ppre_count),
        },

        "total_cached_candidates":
            len(names) * int(args.topk),

        "total_gt_matched_cached_candidates":
            total_gt_matched_cached,

        "duplicate_dense_candidate_count":
            dense_duplicate_candidates,

        "generation_wall_seconds":
            time.time() - start,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "RETENTION_CACHE_BUILD=PASS"
    )


if __name__ == "__main__":
    main()
