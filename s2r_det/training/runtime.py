from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pyarrow.parquet as pq
import torch

from s2r_det.training.prediction_retention import (
    flatten_dense_predictions_batch,
    prediction_retention_loss,
)


def file_names(path):
    return [
        Path(x.strip()).name
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def dataset_name_to_index(dataset):
    out = {}

    for i in range(len(dataset)):
        info = dataset.get_data_info(i)

        name = Path(
            info["img_path"]
        ).name

        if name in out:
            raise RuntimeError(
                f"duplicate dataset filename: {name}"
            )

        out[name] = i

    return out


def failure_image_pools(
    parquet_path,
):
    table = pq.read_table(
        parquet_path,
        columns=[
            "file_name",
            "failure_type",
        ],
    ).to_pydict()

    pools = defaultdict(set)

    for name, ftype in zip(
        table["file_name"],
        table["failure_type"],
    ):
        if ftype not in {
            "F1", "F2", "F3", "F4",
        }:
            continue

        pools[ftype].add(
            Path(name).name
        )

    return {
        k: sorted(v)
        for k, v in pools.items()
    }


def frozen_parameter_sha256(model):
    h = hashlib.sha256()

    for name, p in sorted(
        model.named_parameters(),
        key=lambda x: x[0],
    ):
        if p.requires_grad:
            continue

        h.update(
            name.encode("utf-8")
        )

        h.update(
            p.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return h.hexdigest()


def trainable_snapshot(model):
    return {
        name:
            p.detach()
            .cpu()
            .clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def max_trainable_delta(
    model,
    before,
):
    m = 0.0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        d = float(
            (
                p.detach().cpu()
                - before[name]
            )
            .abs()
            .max()
            .item()
        )

        m = max(m, d)

    return m


def sample_file_name(sample):
    return Path(
        sample.metainfo["img_path"]
    ).name


def batch_retention_loss(
    model,
    raw_head_output,
    data_samples,
    cache_entries,
):
    cls_scores = raw_head_output[0]
    bbox_preds = raw_head_output[1]

    (
        flat_logits,
        flat_boxes,
    ) = flatten_dense_predictions_batch(
        model.bbox_head,
        cls_scores,
        bbox_preds,
        data_samples,
    )

    losses = []
    cls_losses = []
    box_losses = []

    for b, sample in enumerate(
        data_samples
    ):
        name = sample_file_name(
            sample
        )

        if name not in cache_entries:
            raise KeyError(
                f"retention cache missing {name}"
            )

        e = cache_entries[name]

        idx = (
            e["dense_idx"]
            .to(
                device=flat_logits.device,
                dtype=torch.long,
            )
        )

        student_logits = (
            flat_logits[b, idx]
        )

        student_boxes = (
            flat_boxes[b, idx]
        )

        teacher_logits = (
            e["logits"]
            .to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
        )

        teacher_boxes = (
            e["boxes_input"]
            .to(
                device=student_boxes.device,
                dtype=student_boxes.dtype,
            )
        )

        gt_match = (
            e["gt_match"]
            .to(
                device=student_boxes.device,
            )
        )

        ret = prediction_retention_loss(
            student_logits,
            student_boxes,
            teacher_logits,
            teacher_boxes,
            gt_match,
            temperature=2.0,
            beta=1.0,
        )

        losses.append(
            ret["loss_ret"]
        )

        cls_losses.append(
            ret["loss_ret_cls"]
        )

        box_losses.append(
            ret["loss_ret_box"]
        )

    return {
        "loss_ret":
            torch.stack(losses).mean(),

        "loss_ret_cls":
            torch.stack(
                cls_losses
            ).mean(),

        "loss_ret_box":
            torch.stack(
                box_losses
            ).mean(),
    }


def cache_alignment_error(
    model,
    raw_head_output,
    data_samples,
    cache_entries,
):
    """Compare student dense values to frozen teacher after FP16 cast."""

    cls_scores = raw_head_output[0]
    bbox_preds = raw_head_output[1]

    (
        flat_logits,
        flat_boxes,
    ) = flatten_dense_predictions_batch(
        model.bbox_head,
        cls_scores,
        bbox_preds,
        data_samples,
    )

    logit_mismatch = 0
    box_mismatch = 0

    max_logit_abs = 0.0
    max_box_abs = 0.0

    for b, sample in enumerate(
        data_samples
    ):
        name = sample_file_name(
            sample
        )

        e = cache_entries[name]

        idx = (
            e["dense_idx"]
            .to(
                device=flat_logits.device,
                dtype=torch.long,
            )
        )

        slog = (
            flat_logits[b, idx]
            .detach()
            .to(
                dtype=torch.float16,
                device="cpu",
            )
        )

        sbox = (
            flat_boxes[b, idx]
            .detach()
            .to(
                dtype=torch.float16,
                device="cpu",
            )
        )

        tlog = e["logits"]
        tbox = e["boxes_input"]

        if not torch.equal(
            slog,
            tlog,
        ):
            logit_mismatch += 1

        if not torch.equal(
            sbox,
            tbox,
        ):
            box_mismatch += 1

        max_logit_abs = max(
            max_logit_abs,
            float(
                (
                    slog.float()
                    - tlog.float()
                )
                .abs()
                .max()
                .item()
            ),
        )

        max_box_abs = max(
            max_box_abs,
            float(
                (
                    sbox.float()
                    - tbox.float()
                )
                .abs()
                .max()
                .item()
            ),
        )

    return {
        "logit_mismatch_images":
            logit_mismatch,

        "box_mismatch_images":
            box_mismatch,

        "max_logit_abs":
            max_logit_abs,

        "max_box_abs":
            max_box_abs,
    }
