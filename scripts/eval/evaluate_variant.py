#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules

from s2r_det.training.adaptation_variants import (
    configure_variant,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-checkpoint", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--variant", required=True)

    ap.add_argument(
        "--trained-checkpoint",
        default=None,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        args.config
    )

    # Evaluation pipeline/evaluator are inherited from the frozen base.
    dataloader = Runner.build_dataloader(
        cfg.val_dataloader,
        seed=0,
        diff_rank_seed=False,
    )

    controls = json.loads(
        Path(args.controls).read_text()
    )

    model = init_detector(
        args.config,
        args.baseline_checkpoint,
        device="cuda:0",
    )

    meta = configure_variant(
        model,
        args.variant,
        controls,
    )

    if args.trained_checkpoint:
        ckpt = torch.load(
            args.trained_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        state = ckpt.get(
            "state_dict",
            ckpt,
        )

        incompatible = model.load_state_dict(
            state,
            strict=True,
        )

        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "strict checkpoint mismatch"
            )

    model.eval()

    evaluator = Evaluator(
        cfg.val_evaluator
    )

    evaluator.dataset_meta = (
        dataloader.dataset.metainfo
    )

    start = time.time()

    with torch.no_grad():
        for data_batch in dataloader:
            outputs = model.test_step(
                data_batch
            )

            evaluator.process(
                data_samples=outputs,
                data_batch=data_batch,
            )

    metrics = evaluator.evaluate(
        len(dataloader.dataset)
    )

    metrics = {
        str(k): (
            float(v)
            if isinstance(
                v,
                (int, float)
            )
            else v
        )
        for k, v in metrics.items()
    }

    out = {
        "variant":
            args.variant,

        "trained_checkpoint":
            args.trained_checkpoint,

        "validation_images":
            len(dataloader.dataset),

        "trainable_parameters":
            meta[
                "trainable_parameters"
            ],

        "insertion_layers":
            meta[
                "insertion_layers"
            ],

        "metrics":
            metrics,

        "wall_seconds":
            time.time() - start,
    }

    Path(args.output).write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
