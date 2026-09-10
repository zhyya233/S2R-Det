#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from mmengine.config import Config
from mmdet.apis import (
    inference_detector,
    init_detector,
)
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from s2r_det.training.adaptation_variants import (
    configure_variant,
)


def load_b00_eval_module():
    p = Path(
        "scripts/eval/evaluate_baseline.py"
    )

    spec = importlib.util.spec_from_file_location(
        "s2r_b00_eval",
        p,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot import B00 evaluator"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(mod)

    return mod


def resolve_val_coco(config):
    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        config
    )

    ds = DATASETS.build(
        cfg.val_dataloader.dataset
    )

    ds.full_init()

    ann_file = Path(
        ds.ann_file
    )

    if not ann_file.is_file():
        raise FileNotFoundError(
            str(ann_file)
        )

    return ann_file



def standard_coco_metrics(
    coco_path,
    dets,
):
    """COCO standard AP metrics at maxDets=100.

    B00's separate coco_metrics() diagnostic changes maxDets to 500
    in order to expose ARsmall at maxDets=500. COCOeval.stats[0],
    however, is hard-coded to summarize AP at maxDets=100, so it
    becomes -1 when 100 is absent from params.maxDets.

    Formal AP therefore uses an independent standard COCOeval with
    maxDets=[1,10,100]. ARsmall remains the frozen B00 diagnostic
    with maxDets=500.
    """

    gt=COCO(
        str(coco_path)
    )

    dt=(
        gt.loadRes(dets)
        if dets
        else gt.loadRes([])
    )

    ev=COCOeval(
        gt,
        dt,
        "bbox",
    )

    ev.params.maxDets=[
        1,
        10,
        100,
    ]

    ev.evaluate()
    ev.accumulate()

    precision=ev.eval[
        "precision"
    ]

    def mean_valid(x):
        x=x[x>-1]

        if x.size==0:
            return float("nan")

        return float(
            np.mean(x)
        )

    # precision dimensions:
    # [IoU, recall, category, area, maxDets]
    ap=mean_valid(
        precision[
            :,
            :,
            :,
            0,
            2,
        ]
    )

    iou=np.asarray(
        ev.params.iouThrs
    )

    i50=int(
        np.argmin(
            np.abs(
                iou-0.50
            )
        )
    )

    i75=int(
        np.argmin(
            np.abs(
                iou-0.75
            )
        )
    )

    ap50=mean_valid(
        precision[
            i50,
            :,
            :,
            0,
            2,
        ]
    )

    ap75=mean_valid(
        precision[
            i75,
            :,
            :,
            0,
            2,
        ]
    )

    # Standard COCO area labels:
    # 0=all, 1=small, 2=medium, 3=large.
    aps=mean_valid(
        precision[
            :,
            :,
            :,
            1,
            2,
        ]
    )

    return {
        "AP":ap,
        "AP50":ap50,
        "AP75":ap75,
        "APsmall":aps,
        "definition":
            "standard COCO bbox AP; "
            "IoU=.50:.05:.95; "
            "maxDets=100",
    }

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--baseline-checkpoint",
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
        "--trained-checkpoint",
        default=None,
    )

    p.add_argument(
        "--val-root",
        required=True,
    )

    p.add_argument(
        "--toolkit",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--coco-ann",
        default=None,
    )
    p.add_argument(
        "--predictions-output",
        default=None,
    )
    p.add_argument(
        "--test-dev",
        action="store_true",
    )

    a = p.parse_args()

    b00 = load_b00_eval_module()

    vt_status, vt_detail = (
        b00.vt16_unit()
    )

    if vt_status != "PASS":
        raise RuntimeError(
            "VT16 synthetic unit failed: "
            + repr(vt_detail)
        )

    coco_path = (
        Path(a.coco_ann)
        if a.coco_ann
        else resolve_val_coco(a.config)
    )

    if not coco_path.is_file():
        raise FileNotFoundError(str(coco_path))

    with coco_path.open(
        encoding="utf-8"
    ) as f:
        dataset = json.load(f)

    images = sorted(
        dataset["images"],
        key=lambda z:int(z["id"]),
    )

    controls = json.loads(
        Path(
            a.controls
        ).read_text()
    )

    model = init_detector(
        a.config,
        a.baseline_checkpoint,
        device="cuda:0",
    )

    meta = configure_variant(
        model,
        a.variant,
        controls,
    )

    if a.trained_checkpoint:
        ckpt = torch.load(
            a.trained_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        state = ckpt.get(
            "state_dict",
            ckpt,
        )

        incompatible = (
            model.load_state_dict(
                state,
                strict=True,
            )
        )

        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "strict trained-checkpoint mismatch"
            )

    model.eval()

    dets = []
    raw_gt = []
    raw_dt = []
    hs = []
    ws = []

    start = time.time()

    bs = 16

    for st in range(
        0,
        len(images),
        bs,
    ):
        batch = images[
            st:st+bs
        ]

        paths = [
            str(
                Path(a.val_root)
                / "images"
                / im["file_name"]
            )
            for im in batch
        ]

        with torch.no_grad():
            preds = inference_detector(
                model,
                paths,
            )

        if not isinstance(
            preds,
            list,
        ):
            preds = [preds]

        for im,res in zip(
            batch,
            preds,
        ):
            iid = int(im["id"])
            stem = Path(
                im["file_name"]
            ).stem

            hs.append(
                int(im["height"])
            )

            ws.append(
                int(im["width"])
            )

            gt = b00.load_raw_gt(
                Path(a.val_root)
                / "annotations"
                / (stem+".txt")
            )

            raw_gt.append(gt)

            ins = (
                res.pred_instances
                .cpu()
            )

            boxes = (
                ins.bboxes.numpy()
            )

            scores = (
                ins.scores.numpy()
            )

            labels = (
                ins.labels.numpy()
            )

            order = np.argsort(
                -scores
            )

            rows = []

            for j in order:
                x1,y1,x2,y2 = map(
                    float,
                    boxes[j],
                )

                w = max(
                    0.0,
                    x2-x1,
                )

                h = max(
                    0.0,
                    y2-y1,
                )

                score = float(
                    scores[j]
                )

                cat = int(
                    labels[j]
                ) + 1

                rows.append(
                    [
                        x1,y1,w,h,
                        score,cat,
                        -1,-1,
                    ]
                )

                dets.append({
                    "image_id":iid,
                    "category_id":cat,
                    "bbox":[
                        x1,y1,w,h
                    ],
                    "score":score,
                })

            raw_dt.append(
                np.asarray(
                    rows,
                    dtype=np.float64,
                )
                if rows
                else np.zeros(
                    (0,8),
                    dtype=np.float64,
                )
            )

    # Formal COCO AP uses standard maxDets=100.
    coco = standard_coco_metrics(
        coco_path,
        dets,
    )

    # Frozen B00 diagnostic is retained specifically for
    # ARsmall at maxDets=500.
    coco_diagnostic = b00.coco_metrics(
        str(coco_path),
        dets,
    )

    coco["ARsmall"] = float(
        coco_diagnostic["ARsmall"]
    )

    coco["ARsmall_definition"] = (
        "B00 frozen COCO diagnostic; "
        "IoU=.50:.05:.95; "
        "area=small; maxDets=500"
    )

    vt16 = b00.vt16_eval(
        dataset,
        dets,
    )

    official = (
        b00.official_visdrone(
            a.toolkit,
            raw_gt,
            raw_dt,
            hs,
            ws,
        )
    )

    predictions_output_path = None
    if a.predictions_output:
        pred_path = Path(a.predictions_output)
        pred_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pred_path.write_text(
            json.dumps(
                dets,
                ensure_ascii=False,
            )
        )
        predictions_output_path = str(
            pred_path
        )

    result = {
        "variant":
            a.variant,

        "trained_checkpoint":
            a.trained_checkpoint,

        "num_images":
            len(images),

        "prediction_count":
            len(dets),

        "vt16_unit_test":
            vt_status,

        "coco":
            coco,

        "coco_diagnostic":
            coco_diagnostic,

        "vt16":
            vt16,

        "official_visdrone":
            official,

        "model":{
            "total_parameters":
                int(
                    sum(
                        x.numel()
                        for x
                        in model.parameters()
                    )
                ),

            "trainable_parameters":
                int(
                    sum(
                        x.numel()
                        for x
                        in model.parameters()
                        if x.requires_grad
                    )
                ),

            "scope_metadata":
                meta,
        },

        "wall_seconds":
            time.time()-start,

        "evaluator_source":
            "scripts/eval/evaluate_baseline.py",

        "test_dev_used":
            bool(a.test_dev),

        "coco_annotation":
            str(coco_path),

        "predictions_output":
            predictions_output_path,
    }

    Path(
        a.output
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
