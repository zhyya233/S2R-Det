#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from mmengine.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from s2r_det.training.adaptation_variants import configure_variant


ROOT = Path("/home/a/projects/S2R-Det")

CFG = ROOT / "configs/visdrone/rtmdet_tiny_baseline.py"

TEST_ROOT = Path(
    "/home/a/projects/datasets/VisDrone2019-DET/"
    "VisDrone2019-DET-test-dev"
)

ANN = Path(
    "/home/a/projects/datasets/VisDrone2019-DET/"
    "coco_annotations/visdrone2019_det_testdev.json"
)

BASELINE = (
    ROOT
    / "outputs/b00_baseline_corrected/"
      "best_coco_bbox_mAP_epoch_297.pth"
)

FULLFT = (
    ROOT
    / "outputs/x00/X01-EQUALSTEP-FULLFT/train/last.pth"
)

KSTAR = (
    ROOT
    / "outputs/k00/K00-01/train/last.pth"
)

KSTAR_CONTROLS = (
    ROOT
    / "experiments/manifests/k00/K00-01.json"
)

EXPECTED_SHA = {
    "baseline":
        "f0e78b061985f4f33003170afa9d6c74b21301a3f751afe62edbe552a81cf475",
    "equalstep_fullft":
        "01700b4afc4ae862714219e3db233511290569d63737dda49d563d3e6c5163c6",
    "kstar_seed0":
        "d8d52b702b2282064678297789440bccdb6cf74b1270015c11b8c5876879b3e3",
}


CORRUPTIONS = {
    "gaussian_blur": {
        1: {"sigma": 0.8},
        2: {"sigma": 1.6},
        3: {"sigma": 2.4},
    },
    "motion_blur": {
        1: {"length": 5},
        2: {"length": 9},
        3: {"length": 13},
    },
    "jpeg": {
        1: {"quality": 60},
        2: {"quality": 35},
        3: {"quality": 15},
    },
    "gaussian_noise": {
        1: {"sigma_255": 8.0},
        2: {"sigma_255": 16.0},
        3: {"sigma_255": 24.0},
    },
    "low_light": {
        1: {"factor": 0.75},
        2: {"factor": 0.55},
        3: {"factor": 0.35},
    },
    "synthetic_fog": {
        1: {"alpha": 0.12},
        2: {"alpha": 0.25},
        3: {"alpha": 0.40},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def image_path(im):
    name = str(im["file_name"])

    p = TEST_ROOT / "images" / name

    if p.is_file():
        return p

    p = TEST_ROOT / name

    if p.is_file():
        return p

    raise FileNotFoundError(name)


def valid_ann(ann):
    return (
        int(ann.get("iscrowd", 0)) == 0
        and int(ann.get("ignore", 0)) == 0
    )


def density_freeze(data):
    counts = defaultdict(int)

    for ann in data["annotations"]:
        if valid_ann(ann):
            counts[int(ann["image_id"])] += 1

    values = sorted(
        counts.get(int(im["id"]), 0)
        for im in data["images"]
    )

    n = len(values)

    q1 = values[
        int((n - 1) / 3)
    ]

    q2 = values[
        int(2 * (n - 1) / 3)
    ]

    bins = {
        "low": [],
        "medium": [],
        "high": [],
    }

    for im in data["images"]:
        iid = int(im["id"])
        c = counts.get(iid, 0)

        if c <= q1:
            bins["low"].append(iid)
        elif c <= q2:
            bins["medium"].append(iid)
        else:
            bins["high"].append(iid)

    return {
        "definition":
            "tertiles of valid GT objects per test-dev image; "
            "cutpoints frozen before X01/X02 model inference",
        "q1_max_objects": int(q1),
        "q2_max_objects": int(q2),
        "image_counts": {
            k: len(v)
            for k, v in bins.items()
        },
        "bins": bins,
    }


def max_dets_from_config():
    cfg = Config.fromfile(
        str(CFG)
    )

    ev = cfg.get(
        "test_evaluator",
        cfg.get(
            "val_evaluator",
            {},
        ),
    )

    raw = list(
        ev.get(
            "proposal_nums",
            (100, 300, 1000),
        )
    )

    maximum = max(
        int(x)
        for x in raw
    )

    return {
        "config_proposal_nums":
            [int(x) for x in raw],
        "cocoeval_max_dets":
            [1, 10, maximum],
    }


def freeze_protocol(path: Path):
    data = load_json(ANN)

    if len(data["images"]) != 1610:
        raise RuntimeError(
            "test-dev image count mismatch"
        )

    if len(data["annotations"]) != 75102:
        raise RuntimeError(
            "test-dev annotation count mismatch"
        )

    for im in data["images"]:
        image_path(im)

    actual = {
        "baseline": sha256(BASELINE),
        "equalstep_fullft": sha256(FULLFT),
        "kstar_seed0": sha256(KSTAR),
    }

    if actual != EXPECTED_SHA:
        raise RuntimeError(
            f"checkpoint SHA mismatch: {actual}"
        )

    ann_keys = sorted({
        key
        for ann in data["annotations"]
        for key in ann.keys()
    })

    density = density_freeze(data)
    maxd = max_dets_from_config()

    protocol = {
        "stage": "X00",
        "substage": "X01_X02",
        "status":
            "FROZEN_BEFORE_VISDRONE_C_RESULTS",
        "dataset":
            "VisDrone2019-DET test-dev",
        "testdev_images": 1610,
        "testdev_annotations": 75102,
        "models": {
            "baseline": {
                "checkpoint":
                    str(
                        BASELINE.relative_to(
                            ROOT
                        )
                    ),
                "sha256":
                    actual["baseline"],
            },
            "equalstep_fullft": {
                "checkpoint":
                    str(
                        FULLFT.relative_to(
                            ROOT
                        )
                    ),
                "sha256":
                    actual[
                        "equalstep_fullft"
                    ],
                "steps": 42525,
                "seed": 0,
            },
            "kstar_seed0": {
                "checkpoint":
                    str(
                        KSTAR.relative_to(
                            ROOT
                        )
                    ),
                "sha256":
                    actual["kstar_seed0"],
                "variant":
                    "m00_head_r8",
                "steps": 42525,
                "seed": 0,
                "trainable_parameters":
                    3216,
            },
        },
        "visdrone_c": {
            "generation":
                "deterministic on-the-fly; "
                "no corrupted images persisted",
            "corruptions":
                CORRUPTIONS,
            "motion_blur_orientation":
                "deterministic image_id modulo "
                "{0,45,90,135} degrees",
            "gaussian_noise_seed":
                "20260000 + severity*100000 + image_id",
            "report":
                [
                    "AP",
                    "APsmall",
                    "absolute clean drop",
                    "relative clean drop",
                ],
        },
        "x02_visdrone": {
            "category":
                "COCO AP per category",
            "density":
                density[
                    "definition"
                ],
            "density_cutpoints": {
                "low_max":
                    density[
                        "q1_max_objects"
                    ],
                "medium_max":
                    density[
                        "q2_max_objects"
                    ],
            },
            "occlusion":
                "class-aware greedy GT recall "
                "at IoU=0.5 by stored "
                "VisDrone occlusion value",
            "vt16":
                "COCO-style AP and recall "
                "with annotation/detection area "
                "measured after fixed 640 "
                "keep-ratio transform; "
                "VT16 area < 16^2",
        },
        "coco_evaluator":
            maxd,
        "annotation_keys":
            ann_keys,
        "uavdt_x02":
            "reuse frozen G00 sequence "
            "weather/altitude/viewpoint and "
            "GT occlusion diagnostics; "
            "no UAVDT reinference",
        "test_adaptive_use": False,
        "remaining_training": 0,
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
        "X00_PROTOCOL_FREEZE=PASS"
    )
    print(
        "DENSITY_LOW_MAX=",
        density["q1_max_objects"],
    )
    print(
        "DENSITY_MEDIUM_MAX=",
        density["q2_max_objects"],
    )
    print(
        "ANNOTATION_KEYS=",
        ann_keys,
    )


def motion_kernel(length, angle):
    k = np.zeros(
        (length, length),
        dtype=np.float32,
    )

    c = (length - 1) / 2.0
    r = (length - 1) / 2.0

    a = math.radians(angle)

    dx = math.cos(a) * r
    dy = math.sin(a) * r

    p1 = (
        int(round(c - dx)),
        int(round(c - dy)),
    )

    p2 = (
        int(round(c + dx)),
        int(round(c + dy)),
    )

    cv2.line(
        k,
        p1,
        p2,
        1.0,
        thickness=1,
    )

    s = float(k.sum())

    if s <= 0:
        raise RuntimeError(
            "invalid motion kernel"
        )

    return k / s


def corrupt(
    image,
    name,
    severity,
    image_id,
):
    if name == "clean":
        return image

    p = CORRUPTIONS[
        name
    ][severity]

    if name == "gaussian_blur":
        return cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=float(
                p["sigma"]
            ),
            sigmaY=float(
                p["sigma"]
            ),
        )

    if name == "motion_blur":
        angles = (
            0,
            45,
            90,
            135,
        )

        angle = angles[
            int(image_id) % 4
        ]

        kernel = motion_kernel(
            int(p["length"]),
            angle,
        )

        return cv2.filter2D(
            image,
            -1,
            kernel,
        )

    if name == "jpeg":
        ok, buf = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                int(p["quality"]),
            ],
        )

        if not ok:
            raise RuntimeError(
                "JPEG encode failed"
            )

        out = cv2.imdecode(
            buf,
            cv2.IMREAD_COLOR,
        )

        if out is None:
            raise RuntimeError(
                "JPEG decode failed"
            )

        return out

    if name == "gaussian_noise":
        rng = np.random.default_rng(
            20260000
            + int(severity) * 100000
            + int(image_id)
        )

        noise = rng.normal(
            0.0,
            float(
                p["sigma_255"]
            ),
            size=image.shape,
        )

        x = (
            image.astype(
                np.float32
            )
            + noise.astype(
                np.float32
            )
        )

        return np.clip(
            x,
            0,
            255,
        ).astype(
            np.uint8
        )

    if name == "low_light":
        x = (
            image.astype(
                np.float32
            )
            * float(
                p["factor"]
            )
        )

        return np.clip(
            x,
            0,
            255,
        ).astype(
            np.uint8
        )

    if name == "synthetic_fog":
        alpha = float(
            p["alpha"]
        )

        white = np.full_like(
            image,
            255,
        )

        return cv2.addWeighted(
            image,
            1.0 - alpha,
            white,
            alpha,
            0.0,
        )

    raise ValueError(name)


def load_state(
    model,
    path,
):
    x = torch.load(
        path,
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
            f"strict state load failed: {path}"
        )


def build_model(model_id):
    model = init_detector(
        str(CFG),
        str(BASELINE),
        device="cuda:0",
    )

    if model_id == "baseline":
        pass

    elif model_id == "equalstep_fullft":
        load_state(
            model,
            FULLFT,
        )

    elif model_id == "kstar_seed0":
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

    else:
        raise ValueError(
            model_id
        )

    model.eval()

    return model


def predict(
    model,
    images,
    cat_ids,
    corruption_name,
    severity,
    batch_size=16,
):
    image_ids = []
    category_ids = []
    boxes = []
    scores = []

    start = time.time()

    total = len(images)

    for start_idx in range(
        0,
        total,
        batch_size,
    ):
        chunk = images[
            start_idx:
            start_idx + batch_size
        ]

        arrays = []
        ids = []

        for im in chunk:
            iid = int(
                im["id"]
            )

            x = cv2.imread(
                str(
                    image_path(im)
                ),
                cv2.IMREAD_COLOR,
            )

            if x is None:
                raise RuntimeError(
                    f"failed image: {iid}"
                )

            x = corrupt(
                x,
                corruption_name,
                severity,
                iid,
            )

            arrays.append(x)
            ids.append(iid)

        with torch.no_grad():
            outputs = inference_detector(
                model,
                arrays,
            )

        if not isinstance(
            outputs,
            (list, tuple),
        ):
            outputs = [outputs]

        if len(outputs) != len(ids):
            raise RuntimeError(
                "inference batch size mismatch"
            )

        for iid, sample in zip(
            ids,
            outputs,
        ):
            pred = (
                sample.pred_instances
                .to("cpu")
            )

            b = pred.bboxes.numpy()
            s = pred.scores.numpy()
            lab = pred.labels.numpy()

            if len(b) == 0:
                continue

            xywh = np.empty_like(
                b,
                dtype=np.float32,
            )

            xywh[:, 0] = b[:, 0]
            xywh[:, 1] = b[:, 1]
            xywh[:, 2] = (
                b[:, 2]
                - b[:, 0]
            )
            xywh[:, 3] = (
                b[:, 3]
                - b[:, 1]
            )

            mapped = np.asarray([
                cat_ids[int(v)]
                for v in lab
            ], dtype=np.int16)

            image_ids.append(
                np.full(
                    len(b),
                    iid,
                    dtype=np.int32,
                )
            )

            category_ids.append(
                mapped
            )

            boxes.append(
                xywh.astype(
                    np.float32,
                    copy=False,
                )
            )

            scores.append(
                s.astype(
                    np.float32,
                    copy=False,
                )
            )

        batch_no = (
            start_idx // batch_size
            + 1
        )

        if (
            batch_no % 25 == 0
            or start_idx
            + batch_size
            >= total
        ):
            print(
                f"INFER {corruption_name} "
                f"s{severity} "
                f"{min(start_idx + batch_size,total)}/{total}",
                flush=True,
            )

    if image_ids:
        return {
            "image_id":
                np.concatenate(
                    image_ids
                ),
            "category_id":
                np.concatenate(
                    category_ids
                ),
            "bbox":
                np.concatenate(
                    boxes
                ),
            "score":
                np.concatenate(
                    scores
                ),
            "wall_seconds":
                time.time()
                - start,
        }

    return {
        "image_id":
            np.zeros(
                0,
                dtype=np.int32,
            ),
        "category_id":
            np.zeros(
                0,
                dtype=np.int16,
            ),
        "bbox":
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        "score":
            np.zeros(
                0,
                dtype=np.float32,
            ),
        "wall_seconds":
            time.time()
            - start,
    }


def result_list(pred):
    return [
        {
            "image_id":
                int(iid),
            "category_id":
                int(cid),
            "bbox":
                [
                    float(x)
                    for x in box
                ],
            "score":
                float(score),
        }
        for iid, cid, box, score
        in zip(
            pred["image_id"],
            pred["category_id"],
            pred["bbox"],
            pred["score"],
        )
    ]


def load_coco_dt(gt, pred):
    rows = result_list(
        pred
    )

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        dt = gt.loadRes(
            rows
        )

    return dt


def coco_metrics(
    gt,
    pred,
    max_dets,
    img_ids=None,
    cat_ids=None,
):
    dt = load_coco_dt(
        gt,
        pred,
    )

    ev = COCOeval(
        gt,
        dt,
        "bbox",
    )

    ev.params.maxDets = list(
        max_dets
    )

    if img_ids is not None:
        ev.params.imgIds = list(
            img_ids
        )

    if cat_ids is not None:
        ev.params.catIds = list(
            cat_ids
        )

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats

    return {
        "AP":
            float(s[0]),
        "AP50":
            float(s[1]),
        "AP75":
            float(s[2]),
        "APsmall":
            float(s[3]),
        "APmedium":
            float(s[4]),
        "APlarge":
            float(s[5]),
        "AR1":
            float(s[6]),
        "AR10":
            float(s[7]),
        "ARmax":
            float(s[8]),
        "ARsmall":
            float(s[9]),
        "ARmedium":
            float(s[10]),
        "ARlarge":
            float(s[11]),
    }


def scaled_area(
    bbox,
    image,
):
    w = float(
        image["width"]
    )
    h = float(
        image["height"]
    )

    scale = min(
        640.0 / w,
        640.0 / h,
    )

    return (
        float(bbox[2])
        * float(bbox[3])
        * scale
        * scale
    )


def vt16_metrics(
    data,
    pred,
    max_dets,
):
    d = copy.deepcopy(
        data
    )

    images = {
        int(im["id"]): im
        for im in d["images"]
    }

    for ann in d["annotations"]:
        ann["area"] = scaled_area(
            ann["bbox"],
            images[
                int(
                    ann["image_id"]
                )
            ],
        )

    gt = COCO()
    gt.dataset = d

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        gt.createIndex()

    dt = load_coco_dt(
        gt,
        pred,
    )

    for ann in dt.dataset[
        "annotations"
    ]:
        ann["area"] = scaled_area(
            ann["bbox"],
            images[
                int(
                    ann["image_id"]
                )
            ],
        )

    ev = COCOeval(
        gt,
        dt,
        "bbox",
    )

    ev.params.maxDets = list(
        max_dets
    )

    ev.params.areaRng = [
        [0.0, 256.0]
    ]

    ev.params.areaRngLbl = [
        "VT16"
    ]

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        ev.evaluate()
        ev.accumulate()

    p = ev.eval[
        "precision"
    ][:, :, :, 0, -1]

    r = ev.eval[
        "recall"
    ][:, :, 0, -1]

    pvalid = p[
        p > -1
    ]

    rvalid = r[
        r > -1
    ]

    iou50 = int(
        np.argmin(
            np.abs(
                ev.params.iouThrs
                - 0.5
            )
        )
    )

    p50 = p[
        iou50
    ]

    p50 = p50[
        p50 > -1
    ]

    r50 = r[
        iou50
    ]

    r50 = r50[
        r50 > -1
    ]

    return {
        "area_definition":
            "post-640 keep-ratio area < 256",
        "AP":
            float(
                np.mean(pvalid)
            )
            if len(pvalid)
            else 0.0,
        "AP50":
            float(
                np.mean(p50)
            )
            if len(p50)
            else 0.0,
        "AR":
            float(
                np.mean(rvalid)
            )
            if len(rvalid)
            else 0.0,
        "AR50":
            float(
                np.mean(r50)
            )
            if len(r50)
            else 0.0,
    }


def iou_one_to_many(
    box,
    gt_boxes,
):
    if len(gt_boxes) == 0:
        return np.zeros(
            0,
            dtype=np.float64,
        )

    x1 = np.maximum(
        box[0],
        gt_boxes[:, 0],
    )

    y1 = np.maximum(
        box[1],
        gt_boxes[:, 1],
    )

    x2 = np.minimum(
        box[0] + box[2],
        gt_boxes[:, 0]
        + gt_boxes[:, 2],
    )

    y2 = np.minimum(
        box[1] + box[3],
        gt_boxes[:, 1]
        + gt_boxes[:, 3],
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

    a = (
        box[2]
        * box[3]
    )

    ga = (
        gt_boxes[:, 2]
        * gt_boxes[:, 3]
    )

    union = (
        a + ga - inter
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


def occlusion_recall50(
    data,
    pred,
):
    gt = defaultdict(list)

    for ann in data[
        "annotations"
    ]:
        if not valid_ann(ann):
            continue

        occ = ann.get(
            "occlusion",
            ann.get(
                "occluded",
                None,
            ),
        )

        if occ is None:
            continue

        gt[
            (
                int(
                    ann["image_id"]
                ),
                int(
                    ann["category_id"]
                ),
            )
        ].append(
            (
                np.asarray(
                    ann["bbox"],
                    dtype=np.float64,
                ),
                str(occ),
            )
        )

    dt = defaultdict(list)

    for iid, cid, box, score in zip(
        pred["image_id"],
        pred["category_id"],
        pred["bbox"],
        pred["score"],
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

    for key, gt_rows in gt.items():
        boxes = np.stack([
            x[0]
            for x in gt_rows
        ])

        occs = [
            x[1]
            for x in gt_rows
        ]

        used = np.zeros(
            len(boxes),
            dtype=bool,
        )

        for o in occs:
            total[o] += 1

        candidates = sorted(
            dt.get(
                key,
                [],
            ),
            key=lambda x:
                -x[0],
        )

        for _, box in candidates:
            ious = iou_one_to_many(
                box,
                boxes,
            )

            ious[
                used
            ] = -1.0

            j = int(
                np.argmax(
                    ious
                )
            )

            if (
                len(ious)
                and ious[j] >= 0.5
            ):
                used[j] = True
                matched[
                    occs[j]
                ] += 1

    return {
        k: {
            "matched_gt":
                int(
                    matched[k]
                ),
            "total_gt":
                int(total[k]),
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


def clean_diagnostics(
    data,
    coco,
    pred,
    max_dets,
    density,
):
    cat_ids = sorted(
        coco.getCatIds()
    )

    cats = coco.loadCats(
        cat_ids
    )

    cat_name = {
        int(x["id"]):
            str(x["name"])
        for x in cats
    }

    category = {}

    for cid in cat_ids:
        category[
            cat_name[cid]
        ] = coco_metrics(
            coco,
            pred,
            max_dets,
            cat_ids=[cid],
        )

    density_metrics = {}

    for name, ids in density[
        "bins"
    ].items():
        density_metrics[
            name
        ] = {
            "image_count":
                len(ids),
            "metrics":
                coco_metrics(
                    coco,
                    pred,
                    max_dets,
                    img_ids=ids,
                ),
        }

    return {
        "category":
            category,
        "density":
            density_metrics,
        "occlusion_recall50":
            occlusion_recall50(
                data,
                pred,
            ),
        "VT16":
            vt16_metrics(
                data,
                pred,
                max_dets,
            ),
    }


def save_clean_prediction(
    path,
    pred,
):
    np.savez_compressed(
        path,
        image_id=
            pred["image_id"],
        category_id=
            pred["category_id"],
        bbox=
            pred["bbox"],
        score=
            pred["score"],
    )


def reuse_uavdt(out):
    payload = {
        "source_stage": "G00",
        "reinference": False,
        "models": {},
    }

    for mid in [
        "G00-BL",
        "G00-PEFT",
        "G00-S2R",
    ]:
        p = (
            ROOT
            / "outputs/g00/eval_joint"
            / mid
            / "metrics.json"
        )

        x = load_json(p)

        o = x[
            "official_uavdt_source"
        ]

        payload[
            "models"
        ][mid] = {
            "sequence_attributes":
                o[
                    "sequence_attributes"
                ],
            "gt_attribute_recall":
                o[
                    "gt_attribute_recall"
                ],
        }

    (
        out
        / "x02_uavdt_reuse.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
    )

    ap.add_argument(
        "--freeze-protocol",
    )

    ap.add_argument(
        "--model-smoke",
        action="store_true",
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

    if args.model_smoke:
        for mid in [
            "baseline",
            "equalstep_fullft",
            "kstar_seed0",
        ]:
            model = build_model(
                mid
            )

            print(
                "MODEL_SMOKE_PASS",
                mid,
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

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

    density = density_freeze(
        data
    )

    coco = COCO(
        str(ANN)
    )

    cat_ids = sorted(
        coco.getCatIds()
    )

    max_dets = (
        max_dets_from_config()
        ["cocoeval_max_dets"]
    )

    images = sorted(
        data["images"],
        key=lambda x:
            int(x["id"]),
    )

    rows = []
    all_results = {}

    model_ids = [
        "baseline",
        "equalstep_fullft",
        "kstar_seed0",
    ]

    for mid in model_ids:
        print(
            "MODEL_START",
            mid,
            flush=True,
        )

        model = build_model(
            mid
        )

        mdir = (
            out / mid
        )

        mdir.mkdir(
            parents=True,
            exist_ok=False,
        )

        clean = predict(
            model,
            images,
            cat_ids,
            "clean",
            0,
        )

        clean_metrics = (
            coco_metrics(
                coco,
                clean,
                max_dets,
            )
        )

        diagnostics = (
            clean_diagnostics(
                data,
                coco,
                clean,
                max_dets,
                density,
            )
        )

        save_clean_prediction(
            mdir
            / "clean_predictions.npz",
            clean,
        )

        clean_payload = {
            "model_id":
                mid,
            "metrics":
                clean_metrics,
            "diagnostics":
                diagnostics,
            "prediction_count":
                int(
                    len(
                        clean["score"]
                    )
                ),
            "inference_wall_seconds":
                float(
                    clean[
                        "wall_seconds"
                    ]
                ),
        }

        (
            mdir
            / "clean_metrics.json"
        ).write_text(
            json.dumps(
                clean_payload,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        model_results = {
            "clean":
                clean_payload,
            "corruptions":
                {},
        }

        rows.append({
            "model":
                mid,
            "corruption":
                "clean",
            "severity": 0,
            "AP":
                clean_metrics[
                    "AP"
                ],
            "APsmall":
                clean_metrics[
                    "APsmall"
                ],
            "delta_AP":
                0.0,
            "delta_APsmall":
                0.0,
            "relative_AP_drop":
                0.0,
            "relative_APsmall_drop":
                0.0,
            "wall_seconds":
                clean[
                    "wall_seconds"
                ],
        })

        for cname in CORRUPTIONS:
            model_results[
                "corruptions"
            ][cname] = {}

            for sev in [
                1,
                2,
                3,
            ]:
                print(
                    "CONDITION_START",
                    mid,
                    cname,
                    sev,
                    flush=True,
                )

                pred = predict(
                    model,
                    images,
                    cat_ids,
                    cname,
                    sev,
                )

                metrics = coco_metrics(
                    coco,
                    pred,
                    max_dets,
                )

                dap = (
                    metrics["AP"]
                    - clean_metrics["AP"]
                )

                dsmall = (
                    metrics["APsmall"]
                    - clean_metrics[
                        "APsmall"
                    ]
                )

                rel_ap = (
                    -dap
                    / clean_metrics["AP"]
                    if clean_metrics["AP"]
                    else 0.0
                )

                rel_small = (
                    -dsmall
                    / clean_metrics[
                        "APsmall"
                    ]
                    if clean_metrics[
                        "APsmall"
                    ]
                    else 0.0
                )

                item = {
                    "metrics":
                        metrics,
                    "delta_vs_clean": {
                        "AP":
                            float(dap),
                        "APsmall":
                            float(
                                dsmall
                            ),
                    },
                    "relative_drop": {
                        "AP":
                            float(
                                rel_ap
                            ),
                        "APsmall":
                            float(
                                rel_small
                            ),
                    },
                    "prediction_count":
                        int(
                            len(
                                pred[
                                    "score"
                                ]
                            )
                        ),
                    "inference_wall_seconds":
                        float(
                            pred[
                                "wall_seconds"
                            ]
                        ),
                }

                model_results[
                    "corruptions"
                ][cname][
                    str(sev)
                ] = item

                rows.append({
                    "model":
                        mid,
                    "corruption":
                        cname,
                    "severity":
                        sev,
                    "AP":
                        metrics["AP"],
                    "APsmall":
                        metrics[
                            "APsmall"
                        ],
                    "delta_AP":
                        dap,
                    "delta_APsmall":
                        dsmall,
                    "relative_AP_drop":
                        rel_ap,
                    "relative_APsmall_drop":
                        rel_small,
                    "wall_seconds":
                        pred[
                            "wall_seconds"
                        ],
                })

                (
                    mdir
                    / (
                        f"{cname}_"
                        f"s{sev}.json"
                    )
                ).write_text(
                    json.dumps(
                        item,
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

                del pred
                gc.collect()

        (
            mdir
            / "x01_x02_summary.json"
        ).write_text(
            json.dumps(
                model_results,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        all_results[
            mid
        ] = model_results

        del clean
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(
            "MODEL_COMPLETE",
            mid,
            "clean_AP=",
            clean_metrics["AP"],
            "clean_APsmall=",
            clean_metrics[
                "APsmall"
            ],
            flush=True,
        )

    with (
        out
        / "x01_visdronec_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)

    (
        out
        / "x01_x02_all.json"
    ).write_text(
        json.dumps(
            all_results,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    reuse_uavdt(
        out
    )

    print(
        "X01_X02_EVALUATION=COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
