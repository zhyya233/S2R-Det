#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

from s2r_det.training.failure_sampler import (
    FAILURE_TYPES,
    FailureStratifiedBatchSource,
)
from s2r_det.training.runtime import (
    batch_retention_loss,
    dataset_name_to_index,
    failure_image_pools,
    file_names,
    frozen_parameter_sha256,
    max_trainable_delta,
    trainable_snapshot,
)
from s2r_det.training.adaptation_variants import (
    configure_variant,
    freeze_bn_running_stats,
)


def args_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
    )
    p.add_argument(
        "--controls",
        required=True,
    )
    p.add_argument(
        "--variant",
        required=True,
    )
    p.add_argument(
        "--steps",
        type=int,
        required=True,
    )
    p.add_argument(
        "--lr",
        type=float,
        required=True,
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--subset-tsv",
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--formal",
        action="store_true",
    )

    p.add_argument(
        "--failure-parquet",
    )
    p.add_argument(
        "--retention-list",
    )
    p.add_argument(
        "--sampler-manifest",
    )
    p.add_argument(
        "--retention-cache",
    )
    p.add_argument(
        "--retention-lambda",
        type=float,
        default=0.0,
    )

    return p.parse_args()


def main():
    args = args_parser()

    register_all_modules(
        init_default_scope=True
    )

    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )
    random.seed(
        args.seed
    )

    cfg = Config.fromfile(
        args.config
    )

    dataset = DATASETS.build(
        cfg.train_dataloader.dataset
    )
    dataset.full_init()

    # The filename lookup is required only by the failure-aware
    # sampling runtime. Uniform training uses integer dataset indices
    # directly. Keeping this lazy also supports sequence datasets such
    # as UAVDT, whose frame basenames repeat across sequences.
    name_to_index = None

    if args.subset_tsv:
        allowed = [
            int(
                line.split(
                    "\t", 1
                )[0]
            )
            for line in Path(
                args.subset_tsv
            ).read_text().splitlines()
            if line.strip()
        ]
    else:
        allowed = list(
            range(len(dataset))
        )

    model = init_detector(
        args.config,
        args.checkpoint,
        device="cuda:0",
    )

    controls = json.loads(
        Path(
            args.controls
        ).read_text()
    )

    meta = configure_variant(
        model,
        args.variant,
        controls,
    )

    if args.variant == "frozen":
        raise RuntimeError(
            "Frozen is evaluation-only"
        )

    # No batch random resize or other model-side augmentation in P00.
    if hasattr(
        model.data_preprocessor,
        "batch_augments",
    ):
        model.data_preprocessor.batch_augments = None

    model.train()

    if meta[
        "bn_running_stats_frozen"
    ]:
        freeze_bn_running_stats(
            model
        )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if not params:
        raise RuntimeError(
            "no trainable parameters"
        )

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup = min(
        1000,
        max(
            1,
            args.steps // 10,
        ),
    )

    def factor(step):
        if step < warmup:
            start = 1e-5

            return (
                start
                + (1.0 - start)
                * (step + 1)
                / warmup
            )

        remain = max(
            1,
            args.steps - warmup,
        )

        progress = min(
            1.0,
            (
                step
                - warmup
                + 1
            )
            / remain,
        )

        return (
            0.05
            + 0.95
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    scheduler = (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_lambda=factor,
        )
    )

    # --------------------------------------------------------
    # Data source
    # --------------------------------------------------------
    rng = random.Random(
        args.seed
    )

    uses_failure_runtime = (
        args.variant in {
            "s2r_initial",
            "s2r_u2",
            "s2r_u3",
        }
        or (
            args.variant == "m00_head_r8"
            and args.sampler_manifest is not None
        )
    )

    planner = None
    cache_entries = None

    if uses_failure_runtime:
        name_to_index = dataset_name_to_index(
            dataset
        )

        required = {
            "--failure-parquet":
                args.failure_parquet,
            "--retention-list":
                args.retention_list,
            "--sampler-manifest":
                args.sampler_manifest,
            "--retention-cache":
                args.retention_cache,
        }

        missing = [
            k
            for k, v in required.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "missing S2R resources: "
                + ",".join(missing)
            )

        sm = json.loads(
            Path(
                args.sampler_manifest
            ).read_text()
        )

        pools = failure_image_pools(
            args.failure_parquet
        )

        retention_names = file_names(
            args.retention_list
        )

        planner = (
            FailureStratifiedBatchSource(
                failure_pools=pools,
                retention_pool=
                    retention_names,
                failure_quotas=
                    sm[
                        "failure_type_quotas"
                    ],
                batch_size=
                    args.batch_size,
                seed=args.seed,
                failure_batch_fraction=
                    float(
                        sm.get(
                            "source_policy",
                            {},
                        ).get(
                            "failure_batch_fraction",
                            0.70,
                        )
                    ),
            )
        )

        cache_payload = torch.load(
            args.retention_cache,
            map_location="cpu",
            weights_only=True,
        )

        cache_entries = (
            cache_payload["entries"]
        )

        planned = list(
            planner.iter_batches(
                args.steps
            )
        )
    else:
        planned = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    before_frozen = (
        frozen_parameter_sha256(
            model
        )
    )

    before_trainable = (
        trainable_snapshot(
            model
        )
    )

    captures = {}

    def hook(
        module,
        inputs,
        output,
    ):
        captures["head"] = output

    handle = (
        model.bbox_head
        .register_forward_hook(
            hook
        )
    )

    source_counts = Counter()
    losses = []
    det_losses = []
    ret_losses = []
    ret_cls_losses = []
    ret_box_losses = []

    peak = 0
    start = time.time()

    out = Path(
        args.output
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        for step in range(
            args.steps
        ):
            if planned is None:
                idxs = [
                    allowed[
                        rng.randrange(
                            len(allowed)
                        )
                    ]
                    for _ in range(
                        args.batch_size
                    )
                ]

                source = "uniform"

            else:
                spec = planned[
                    step
                ]

                source = spec[
                    "source"
                ]

                names = spec[
                    "file_names"
                ]

                try:
                    idxs = [
                        name_to_index[
                            name
                        ]
                        for name in names
                    ]
                except KeyError as e:
                    raise RuntimeError(
                        "sampled filename absent "
                        "from training dataset: "
                        + str(e)
                    )

            source_counts[
                source
            ] += 1

            batch = pseudo_collate(
                [
                    dataset[i]
                    for i in idxs
                ]
            )

            processed = (
                model.data_preprocessor(
                    batch,
                    training=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            captures.clear()

            loss_dict = (
                model._run_forward(
                    processed,
                    mode="loss",
                )
            )

            det_loss, _ = (
                model.parse_losses(
                    loss_dict
                )
            )

            ret_loss = (
                det_loss * 0.0
            )
            ret_cls = (
                det_loss * 0.0
            )
            ret_box = (
                det_loss * 0.0
            )

            if (
                uses_failure_runtime
                and source == "retention"
                and args.retention_lambda > 0.0
            ):
                if "head" not in captures:
                    raise RuntimeError(
                        "bbox_head hook missing"
                    )

                ret = batch_retention_loss(
                    model,
                    captures["head"],
                    processed[
                        "data_samples"
                    ],
                    cache_entries,
                )

                ret_loss = ret[
                    "loss_ret"
                ]

                ret_cls = ret[
                    "loss_ret_cls"
                ]

                ret_box = ret[
                    "loss_ret_box"
                ]

            total_loss = (
                det_loss
                + float(
                    args.retention_lambda
                )
                * ret_loss
            )

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    f"nonfinite total loss "
                    f"at step {step}"
                )

            total_loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(
                float(
                    total_loss
                    .detach()
                    .cpu()
                )
            )

            det_losses.append(
                float(
                    det_loss
                    .detach()
                    .cpu()
                )
            )

            ret_losses.append(
                float(
                    ret_loss
                    .detach()
                    .cpu()
                )
            )

            ret_cls_losses.append(
                float(
                    ret_cls
                    .detach()
                    .cpu()
                )
            )

            ret_box_losses.append(
                float(
                    ret_box
                    .detach()
                    .cpu()
                )
            )

            peak = max(
                peak,
                int(
                    torch.cuda
                    .max_memory_allocated()
                    / 1024
                    / 1024
                ),
            )

            if (
                step == 0
                or (step + 1) % 100 == 0
                or step + 1 == args.steps
            ):
                print(
                    f"step={step+1}|"
                    f"source={source}|"
                    f"det={det_losses[-1]:.8f}|"
                    f"ret={ret_losses[-1]:.8f}|"
                    f"total={losses[-1]:.8f}|"
                    f"lr={optimizer.param_groups[0]['lr']:.10g}",
                    flush=True,
                )

    finally:
        handle.remove()

    after_frozen = (
        frozen_parameter_sha256(
            model
        )
    )

    delta = max_trainable_delta(
        model,
        before_trainable,
    )

    checkpoint = {
        "state_dict":
            model.state_dict(),

        "meta": {
            "variant":
                args.variant,

            "steps":
                args.steps,

            "lr":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "seed":
                args.seed,

            "retention_lambda":
                args.retention_lambda,

            "trainable_parameters":
                meta[
                    "trainable_parameters"
                ],

            "insertion_layers":
                meta[
                    "insertion_layers"
                ],
        },
    }

    torch.save(
        checkpoint,
        out / "last.pth",
    )

    summary = {
        **meta,

        "steps":
            args.steps,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "seed":
            args.seed,

        "retention_lambda":
            args.retention_lambda,

        "source_counts":
            dict(
                source_counts
            ),

        "wall_time_hours":
            (
                time.time()
                - start
            )
            / 3600,

        "peak_vram_mb":
            peak,

        "initial_loss":
            losses[0],

        "final_loss":
            losses[-1],

        "min_loss":
            min(losses),

        "all_finite":
            all(
                math.isfinite(x)
                for x in losses
            ),

        "mean_det_loss":
            sum(det_losses)
            / len(det_losses),

        "mean_ret_loss":
            sum(ret_losses)
            / len(ret_losses),

        "max_ret_loss":
            max(ret_losses),

        "max_ret_cls_loss":
            max(
                ret_cls_losses
            ),

        "max_ret_box_loss":
            max(
                ret_box_losses
            ),

        "frozen_parameter_hash_same":
            before_frozen
            == after_frozen,

        "max_trainable_parameter_delta":
            delta,
    }

    (
        out
        / "train_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
