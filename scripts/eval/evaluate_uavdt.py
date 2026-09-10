#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import gzip
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from s2r_det.training.adaptation_variants import configure_variant


ROOT = Path("/home/a/projects/S2R-Det")
UAV = Path("/home/a/projects/datasets/UAVDT")

CFG = ROOT / "configs/uavdt/rtmdet_tiny_baseline.py"
TEST_JSON = UAV / "derived/coco/test.json"

GTROOT = (
    UAV
    / "raw/UAV-benchmark-MOTD_v1.0/GT"
)

BL = ROOT / "outputs/g00/G00-BL/epoch_300.pth"
PEFT = ROOT / "outputs/g00/G00-PEFT/train/last.pth"
S2R = ROOT / "outputs/g00/G00-S2R/train/last.pth"

PEFT_CONTROLS = ROOT / "analysis/p00/frozen_controls.json"
S2R_CONTROLS = ROOT / "experiments/manifests/g00/G00-S2R.json"

SEQUENCES = [
    "M0203", "M0205", "M0208", "M0209",
    "M0403", "M0601", "M0602", "M0606",
    "M0701", "M0801", "M0802", "M1001",
    "M1004", "M1007", "M1009", "M1101",
    "M1301", "M1302", "M1303", "M1401",
]

SEQ_LENGTHS = [
    1007, 646, 265, 1576,
    514, 372, 480, 1374,
    1308, 298, 1101, 1859,
    269, 659, 604, 864,
    1182, 719, 445, 1050,
]

IGNORE_PASS = {
    "M0203", "M0205", "M0208",
    "M0403", "M0601", "M0602", "M0606",
    "M0701", "M0802",
    "M1001", "M1004", "M1007", "M1009",
    "M1101",
    "M1301", "M1302", "M1303",
    "M1401",
}

CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "bus",
}

SEQ_ATTR_KEYS = {
    "weather": [
        "daylight",
        "night",
        "fog",
    ],
    "altitude": [
        "low_alt",
        "medium_alt",
        "high_alt",
    ],
    "viewpoint": [
        "front_view",
        "side_view",
        "bird_view",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()


def load_txt(path: Path, cols=None):
    if not path.is_file() or path.stat().st_size == 0:
        n = len(cols) if cols is not None else 0
        return np.zeros((0, n), dtype=np.float64)

    kw = dict(
        delimiter=",",
        ndmin=2,
        dtype=np.float64,
    )

    if cols is not None:
        kw["usecols"] = cols

    x = np.loadtxt(path, **kw)
    return np.asarray(x, dtype=np.float64)


def load_dataset_metadata():
    x = json.loads(
        TEST_JSON.read_text(
            encoding="utf-8"
        )
    )

    id_to_seq_frame = {}
    frame_to_img = {}
    seq_attrs = {}

    for im in x["images"]:
        iid = int(im["id"])
        seq = str(im["sequence_id"])
        fr = int(im["frame_index"])

        id_to_seq_frame[iid] = (
            seq,
            fr,
        )

        frame_to_img[(seq, fr)] = iid

        attrs = dict(
            im.get("attributes", {})
        )

        if seq in seq_attrs:
            if seq_attrs[seq] != attrs:
                raise RuntimeError(
                    f"inconsistent sequence attributes: {seq}"
                )
        else:
            seq_attrs[seq] = attrs

    if set(seq_attrs) != set(SEQUENCES):
        raise RuntimeError(
            "derived test sequences differ from official list"
        )

    expected = dict(
        zip(SEQUENCES, SEQ_LENGTHS)
    )

    for seq, n in expected.items():
        got = sum(
            1
            for s, _ in id_to_seq_frame.values()
            if s == seq
        )

        if got != n:
            raise RuntimeError(
                f"{seq}: expected {n} frames, got {got}"
            )

    return (
        x,
        id_to_seq_frame,
        frame_to_img,
        seq_attrs,
    )


def load_gt():
    by_seq = {}
    ignore = {}

    for seq in SEQUENCES:
        gt = load_txt(
            GTROOT / f"{seq}_gt_whole.txt"
        )

        if gt.shape[1] < 9:
            raise RuntimeError(
                f"{seq}: bad whole GT shape {gt.shape}"
            )

        frame_map = defaultdict(list)

        for row in gt:
            frame_map[int(row[0])].append(
                row.copy()
            )

        by_seq[seq] = {
            fr: np.asarray(
                rows,
                dtype=np.float64,
            )
            for fr, rows in frame_map.items()
        }

        ign_path = (
            GTROOT
            / f"{seq}_gt_ignore.txt"
        )

        ign = load_txt(ign_path)

        ign_map = defaultdict(list)

        if ign.size:
            for row in ign:
                ign_map[int(row[0])].append(
                    row.copy()
                )

        ignore[seq] = {
            fr: np.asarray(
                rows,
                dtype=np.float64,
            )
            for fr, rows in ign_map.items()
        }

    return by_seq, ignore


def filter_external_ignore(
    det,
    ign_rows,
):
    if det.size == 0:
        return det

    if ign_rows is None or len(ign_rows) == 0:
        return det

    keep = np.ones(
        len(det),
        dtype=bool,
    )

    x = det[:, 0]
    y = det[:, 1]
    w = det[:, 2]
    h = det[:, 3]

    for r in ign_rows:
        ix = float(r[2])
        iy = float(r[3])
        iw = float(r[4])
        ih = float(r[5])

        inside = (
            (x > ix)
            & (y > iy)
            & ((x + w) < (ix + iw))
            & ((y + h) < (iy + ih))
        )

        keep &= ~inside

    return det[keep]


def iou_matrix(dt, gt):
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros(
            (len(dt), len(gt)),
            dtype=np.float64,
        )

    dx1 = dt[:, 0][:, None]
    dy1 = dt[:, 1][:, None]
    dx2 = (
        dt[:, 0] + dt[:, 2]
    )[:, None]
    dy2 = (
        dt[:, 1] + dt[:, 3]
    )[:, None]

    gx1 = gt[:, 0][None, :]
    gy1 = gt[:, 1][None, :]
    gx2 = (
        gt[:, 0] + gt[:, 2]
    )[None, :]
    gy2 = (
        gt[:, 1] + gt[:, 3]
    )[None, :]

    iw = np.maximum(
        0.0,
        np.minimum(dx2, gx2)
        - np.maximum(dx1, gx1),
    )

    ih = np.maximum(
        0.0,
        np.minimum(dy2, gy2)
        - np.maximum(dy1, gy1),
    )

    inter = iw * ih

    da = (
        dt[:, 2] * dt[:, 3]
    )[:, None]

    ga = (
        gt[:, 2] * gt[:, 3]
    )[None, :]

    union = da + ga - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def eval_frame(
    gt_rows,
    det_rows,
    thr,
):
    if gt_rows is None:
        gt_rows = np.zeros(
            (0, 9),
            dtype=np.float64,
        )

    if det_rows is None:
        det_rows = np.zeros(
            (0, 5),
            dtype=np.float64,
        )

    if len(det_rows):
        order = np.argsort(
            -det_rows[:, 4],
            kind="stable",
        )
        det_rows = det_rows[order]

    gt_boxes = (
        gt_rows[:, 2:6]
        if len(gt_rows)
        else np.zeros(
            (0, 4),
            dtype=np.float64,
        )
    )

    dt_boxes = (
        det_rows[:, :4]
        if len(det_rows)
        else np.zeros(
            (0, 4),
            dtype=np.float64,
        )
    )

    oa = iou_matrix(
        dt_boxes,
        gt_boxes,
    )

    gt_match = np.zeros(
        len(gt_rows),
        dtype=np.int8,
    )

    dt_match = np.zeros(
        len(det_rows),
        dtype=np.int8,
    )

    for d in range(len(det_rows)):
        best_oa = float(thr)
        best_g = -1

        for g in range(len(gt_rows)):
            if gt_match[g] == 1:
                continue

            v = float(oa[d, g])

            # MATLAB evalRes:
            # if(oa(d,g)<bstOa), continue; end
            if v < best_oa:
                continue

            best_oa = v
            best_g = g

        if best_g >= 0:
            gt_match[best_g] = 1
            dt_match[d] = 1

    return (
        gt_match,
        det_rows[:, 4].copy(),
        dt_match,
    )


def voc_ap(
    scores,
    matches,
    num_gt,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    matches = np.asarray(
        matches,
        dtype=np.int8,
    )

    if len(scores) == 0:
        return {
            "num_gt": int(num_gt),
            "num_detections": 0,
            "ap": 0.0,
            "ap_pct": 0.0,
            "ap_pct_round2": 0.0,
            "final_recall": 0.0,
        }

    order = np.argsort(
        -scores,
        kind="stable",
    )

    m = matches[order]

    fp = np.cumsum(
        m == 0
    )

    tp = np.cumsum(
        m == 1
    )

    rec = (
        tp
        / max(1, int(num_gt))
    )

    prec = (
        tp
        / np.maximum(
            1,
            fp + tp,
        )
    )

    mrec = np.concatenate(
        ([0.0], rec, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], prec, [0.0])
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1,
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1],
        )

    idx = np.where(
        mrec[1:] != mrec[:-1]
    )[0] + 1

    ap = float(
        np.sum(
            (
                mrec[idx]
                - mrec[idx - 1]
            )
            * mpre[idx]
        )
    )

    return {
        "num_gt": int(num_gt),
        "num_detections": int(len(scores)),
        "ap": ap,
        "ap_pct": ap * 100.0,
        "ap_pct_round2": round(
            ap * 100.0,
            2,
        ),
        "final_recall": (
            float(rec[-1])
            if len(rec)
            else 0.0
        ),
    }


def build_model(
    model_id,
):
    if model_id == "G00-BL":
        model = init_detector(
            str(CFG),
            str(BL),
            device="cuda:0",
        )
        meta = {
            "variant": "baseline",
            "trainable_parameters": 0,
            "insertion_layers": [],
        }

    elif model_id == "G00-PEFT":
        model = init_detector(
            str(CFG),
            str(BL),
            device="cuda:0",
        )

        controls = json.loads(
            PEFT_CONTROLS.read_text(
                encoding="utf-8"
            )
        )

        meta = configure_variant(
            model,
            "conv_adapter",
            controls,
        )

        x = torch.load(
            PEFT,
            map_location="cpu",
            weights_only=True,
        )

        state = x.get(
            "state_dict",
            x,
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
                "G00-PEFT strict load failed"
            )

    elif model_id == "G00-S2R":
        model = init_detector(
            str(CFG),
            str(BL),
            device="cuda:0",
        )

        controls = json.loads(
            S2R_CONTROLS.read_text(
                encoding="utf-8"
            )
        )

        meta = configure_variant(
            model,
            "m00_head_r8",
            controls,
        )

        x = torch.load(
            S2R,
            map_location="cpu",
            weights_only=True,
        )

        state = x.get(
            "state_dict",
            x,
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
                "G00-S2R strict load failed"
            )

    else:
        raise ValueError(model_id)

    model.eval()

    return model, meta


def run_inference(
    model_id,
    output_dir,
):
    cfg = Config.fromfile(
        str(CFG)
    )

    loader = Runner.build_dataloader(
        cfg.test_dataloader,
        seed=0,
        diff_rank_seed=False,
    )

    model, meta = build_model(
        model_id
    )

    img_ids = []
    cat_ids = []
    boxes = []
    scores = []

    start = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(
            loader,
            start=1,
        ):
            outputs = model.test_step(
                batch
            )

            for sample in outputs:
                iid = int(
                    sample.metainfo["img_id"]
                )

                pred = (
                    sample.pred_instances
                    .to("cpu")
                )

                b = (
                    pred.bboxes
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                s = (
                    pred.scores
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                lab = (
                    pred.labels
                    .numpy()
                    .astype(
                        np.int16,
                        copy=False,
                    )
                )

                if len(b) == 0:
                    continue

                xywh = np.empty_like(
                    b,
                    dtype=np.float32,
                )

                xywh[:, 0] = b[:, 0]
                xywh[:, 1] = b[:, 1]
                xywh[:, 2] = (
                    b[:, 2] - b[:, 0]
                )
                xywh[:, 3] = (
                    b[:, 3] - b[:, 1]
                )

                img_ids.append(
                    np.full(
                        len(b),
                        iid,
                        dtype=np.int32,
                    )
                )

                cat_ids.append(
                    lab + 1
                )

                boxes.append(xywh)
                scores.append(s)

            if bi % 100 == 0:
                print(
                    f"{model_id}: batch "
                    f"{bi}/{len(loader)}",
                    flush=True,
                )

    if img_ids:
        image_id = np.concatenate(
            img_ids
        )
        category_id = np.concatenate(
            cat_ids
        )
        bbox = np.concatenate(
            boxes
        )
        score = np.concatenate(
            scores
        )
    else:
        image_id = np.zeros(
            0,
            dtype=np.int32,
        )
        category_id = np.zeros(
            0,
            dtype=np.int16,
        )
        bbox = np.zeros(
            (0, 4),
            dtype=np.float32,
        )
        score = np.zeros(
            0,
            dtype=np.float32,
        )

    wall = time.time() - start

    pred_path = (
        output_dir
        / model_id
        / "predictions.npz"
    )

    pred_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        pred_path,
        image_id=image_id,
        category_id=category_id,
        bbox=bbox,
        score=score,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
        "inference_wall_seconds": wall,
        "model_meta": meta,
        "prediction_file":
            str(pred_path),
    }


def coco_metrics(
    prediction,
):
    gt = COCO(
        str(TEST_JSON)
    )

    n = len(
        prediction["score"]
    )

    if n:
        arr = np.column_stack([
            prediction["image_id"],
            prediction["bbox"],
            prediction["score"],
            prediction["category_id"],
        ]).astype(
            np.float64,
            copy=False,
        )
    else:
        arr = np.zeros(
            (0, 7),
            dtype=np.float64,
        )

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        dt = gt.loadRes(arr)

    def evaluate(cat_ids=None):
        ev = COCOeval(
            gt,
            dt,
            "bbox",
        )

        ev.params.maxDets = [
            1,
            10,
            100,
        ]

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
            "AP": float(s[0]),
            "AP50": float(s[1]),
            "AP75": float(s[2]),
            "APsmall": float(s[3]),
            "APmedium": float(s[4]),
            "APlarge": float(s[5]),
            "AR1": float(s[6]),
            "AR10": float(s[7]),
            "AR100": float(s[8]),
            "ARsmall": float(s[9]),
            "ARmedium": float(s[10]),
            "ARlarge": float(s[11]),
        }

    overall = evaluate()

    per_category = {}

    for cid, name in CATEGORY_NAMES.items():
        per_category[name] = evaluate(
            [cid]
        )

    return {
        "overall": overall,
        "per_category": per_category,
    }


def official_metrics(
    prediction,
    id_to_seq_frame,
    frame_to_img,
    seq_attrs,
    gt_by_seq,
    ignore_by_seq,
):
    det_by_img = defaultdict(list)

    for iid, box, score in zip(
        prediction["image_id"],
        prediction["bbox"],
        prediction["score"],
    ):
        det_by_img[int(iid)].append(
            [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(score),
            ]
        )

    det_by_img = {
        iid: np.asarray(
            rows,
            dtype=np.float64,
        )
        for iid, rows
        in det_by_img.items()
    }

    thresholds = [
        0.7,
        0.5,
    ]

    seq_eval = {
        thr: {
            seq: {
                "scores": [],
                "matches": [],
                "num_gt": 0,
            }
            for seq in SEQUENCES
        }
        for thr in thresholds
    }

    recall_counters = {
        thr: {
            "category": defaultdict(
                lambda: [0, 0]
            ),
            "occlusion": defaultdict(
                lambda: [0, 0]
            ),
            "out_of_view": defaultdict(
                lambda: [0, 0]
            ),
        }
        for thr in thresholds
    }

    ignored_detection_count = 0

    for seq, seq_len in zip(
        SEQUENCES,
        SEQ_LENGTHS,
    ):
        for fr in range(
            1,
            seq_len + 1,
        ):
            iid = frame_to_img[
                (seq, fr)
            ]

            det = det_by_img.get(
                iid,
                np.zeros(
                    (0, 5),
                    dtype=np.float64,
                ),
            )

            before = len(det)

            if seq in IGNORE_PASS:
                det = filter_external_ignore(
                    det,
                    ignore_by_seq[
                        seq
                    ].get(fr),
                )

            ignored_detection_count += (
                before - len(det)
            )

            gt = gt_by_seq[
                seq
            ].get(
                fr,
                np.zeros(
                    (0, 9),
                    dtype=np.float64,
                ),
            )

            for thr in thresholds:
                (
                    gt_match,
                    scores,
                    dt_match,
                ) = eval_frame(
                    gt,
                    det,
                    thr,
                )

                q = seq_eval[
                    thr
                ][seq]

                q["scores"].extend(
                    scores.tolist()
                )

                q["matches"].extend(
                    dt_match.tolist()
                )

                q["num_gt"] += len(gt)

                for gi, row in enumerate(gt):
                    matched = int(
                        gt_match[gi] == 1
                    )

                    cat = int(row[6])
                    occ = int(row[7])
                    outv = int(row[8])

                    for family, key in [
                        ("category", cat),
                        ("occlusion", occ),
                        ("out_of_view", outv),
                    ]:
                        rec = (
                            recall_counters[
                                thr
                            ][family][key]
                        )

                        rec[0] += matched
                        rec[1] += 1

    def aggregate(
        thr,
        seqs,
    ):
        scores = []
        matches = []
        num_gt = 0

        for seq in seqs:
            q = seq_eval[
                thr
            ][seq]

            scores.extend(
                q["scores"]
            )

            matches.extend(
                q["matches"]
            )

            num_gt += int(
                q["num_gt"]
            )

        return voc_ap(
            scores,
            matches,
            num_gt,
        )

    result = {
        "source_faithful_official": {
            "iou_threshold": 0.7,
            "overall":
                aggregate(
                    0.7,
                    SEQUENCES,
                ),
        },
        "route_compatibility": {
            "iou_threshold": 0.5,
            "overall":
                aggregate(
                    0.5,
                    SEQUENCES,
                ),
        },
        "sequence_attributes": {},
        "gt_attribute_recall": {},
        "ignored_detection_count":
            int(
                ignored_detection_count
            ),
    }

    for family, keys in (
        SEQ_ATTR_KEYS.items()
    ):
        result[
            "sequence_attributes"
        ][family] = {}

        for key in keys:
            selected = [
                seq
                for seq in SEQUENCES
                if int(
                    seq_attrs[
                        seq
                    ].get(
                        key,
                        0,
                    )
                ) == 1
            ]

            result[
                "sequence_attributes"
            ][family][key] = {
                "sequences": selected,
                "ap70":
                    aggregate(
                        0.7,
                        selected,
                    ),
                "ap50":
                    aggregate(
                        0.5,
                        selected,
                    ),
            }

    for thr in thresholds:
        tname = (
            "iou70"
            if thr == 0.7
            else "iou50"
        )

        result[
            "gt_attribute_recall"
        ][tname] = {}

        for family in [
            "category",
            "occlusion",
            "out_of_view",
        ]:
            result[
                "gt_attribute_recall"
            ][tname][family] = {}

            for key, (
                matched,
                total,
            ) in sorted(
                recall_counters[
                    thr
                ][family].items()
            ):
                if family == "category":
                    name = (
                        CATEGORY_NAMES.get(
                            int(key),
                            str(key),
                        )
                    )
                else:
                    name = str(key)

                result[
                    "gt_attribute_recall"
                ][tname][family][name] = {
                    "matched_gt":
                        int(matched),
                    "total_gt":
                        int(total),
                    "recall":
                        (
                            float(
                                matched
                                / total
                            )
                            if total
                            else 0.0
                        ),
                }

    return result


def save_prediction_json_gz(
    prediction,
    path,
):
    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
    ) as f:
        for iid, cid, box, score in zip(
            prediction["image_id"],
            prediction["category_id"],
            prediction["bbox"],
            prediction["score"],
        ):
            row = {
                "image_id": int(iid),
                "category_id": int(cid),
                "bbox": [
                    float(x)
                    for x in box
                ],
                "score": float(score),
            }

            f.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    register_all_modules(
        init_default_scope=True
    )

    (
        dataset_json,
        id_to_seq_frame,
        frame_to_img,
        seq_attrs,
    ) = load_dataset_metadata()

    gt_by_seq, ignore_by_seq = (
        load_gt()
    )

    expected_shas = {
        "G00-BL":
            "f66e04567c6730d74fdfdb0ff856cc9806f5c9fae20bdf6a12ac2065106f5989",
        "G00-PEFT":
            "0172506e6a95db8f125fb59315c8c8b8451363b38ee78f92995af61bf5876a51",
        "G00-S2R":
            "0dd8ac5801410b136bda37d7cd8ab493ad5bff8bb3826de2a4e3d664e22c8b77",
    }

    actual_shas = {
        "G00-BL": sha256(BL),
        "G00-PEFT": sha256(PEFT),
        "G00-S2R": sha256(S2R),
    }

    if actual_shas != expected_shas:
        raise RuntimeError(
            "frozen checkpoint SHA mismatch"
        )

    if args.dry_run:
        for mid in [
            "G00-BL",
            "G00-PEFT",
            "G00-S2R",
        ]:
            model, meta = build_model(
                mid
            )

            print(
                "DRY_MODEL_PASS",
                mid,
                meta.get(
                    "trainable_parameters",
                    0,
                ),
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

        print(
            "DRY_RUN=PASS",
            "test_images=",
            len(
                dataset_json["images"]
            ),
            "test_annotations=",
            len(
                dataset_json[
                    "annotations"
                ]
            ),
        )

        return

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    protocol = {
        "dataset": "UAVDT",
        "test_images":
            len(
                dataset_json[
                    "images"
                ]
            ),
        "test_annotations":
            len(
                dataset_json[
                    "annotations"
                ]
            ),
        "checkpoint_sha256":
            actual_shas,
        "official_source_semantics": {
            "overall_is_class_agnostic":
                True,
            "evalRes_default_iou":
                0.7,
            "ignore_policy":
                "remove detections strictly fully contained "
                "inside official ignore boxes for the official "
                "pass sequence list",
            "ap":
                "VOC precision-envelope integral",
        },
        "route_compatibility_metric": {
            "iou": 0.5,
            "note":
                "retained because the preregistered project "
                "route incorrectly described official UAVDT "
                "AP as AP@0.5; it is not labeled official",
        },
        "viewpoint_is_multilabel":
            True,
        "test_adaptive_use":
            False,
    }

    (
        output
        / "evaluation_protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = {}

    model_params = {
        "G00-BL": 0,
        "G00-PEFT": 21600,
        "G00-S2R": 3216,
    }

    for model_id in [
        "G00-BL",
        "G00-PEFT",
        "G00-S2R",
    ]:
        print(
            "MODEL_START",
            model_id,
            flush=True,
        )

        pred = run_inference(
            model_id,
            output,
        )

        model_dir = (
            output / model_id
        )

        save_prediction_json_gz(
            pred,
            model_dir
            / "predictions.jsonl.gz",
        )

        official = official_metrics(
            pred,
            id_to_seq_frame,
            frame_to_img,
            seq_attrs,
            gt_by_seq,
            ignore_by_seq,
        )

        coco = coco_metrics(
            pred
        )

        metrics = {
            "experiment_id":
                model_id,
            "checkpoint_sha256":
                actual_shas[
                    model_id
                ],
            "adaptation_trainable_parameters":
                model_params[
                    model_id
                ],
            "num_predictions":
                int(
                    len(
                        pred["score"]
                    )
                ),
            "inference_wall_seconds":
                float(
                    pred[
                        "inference_wall_seconds"
                    ]
                ),
            "official_uavdt_source":
                official,
            "coco":
                coco,
        }

        (
            model_dir
            / "metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        summaries[
            model_id
        ] = metrics

        print(
            "MODEL_COMPLETE",
            model_id,
            "official_AP70=",
            official[
                "source_faithful_official"
            ][
                "overall"
            ][
                "ap_pct_round2"
            ],
            "route_AP50=",
            official[
                "route_compatibility"
            ][
                "overall"
            ][
                "ap_pct_round2"
            ],
            "COCO_AP=",
            coco[
                "overall"
            ][
                "AP"
            ],
            flush=True,
        )

        del pred
        gc.collect()
        torch.cuda.empty_cache()

    base = summaries[
        "G00-BL"
    ]

    comparison = {
        "metric_naming": {
            "official":
                "source-faithful UAVDT VOC AP@0.7",
            "route_compat":
                "project-route compatibility VOC AP@0.5",
        },
        "baseline":
            "G00-BL",
        "methods": {},
    }

    b70 = base[
        "official_uavdt_source"
    ][
        "source_faithful_official"
    ][
        "overall"
    ][
        "ap"
    ]

    b50 = base[
        "official_uavdt_source"
    ][
        "route_compatibility"
    ][
        "overall"
    ][
        "ap"
    ]

    bcap = base[
        "coco"
    ][
        "overall"
    ][
        "AP"
    ]

    rows = []

    for mid, m in summaries.items():
        a70 = m[
            "official_uavdt_source"
        ][
            "source_faithful_official"
        ][
            "overall"
        ][
            "ap"
        ]

        a50 = m[
            "official_uavdt_source"
        ][
            "route_compatibility"
        ][
            "overall"
        ][
            "ap"
        ]

        cap = m[
            "coco"
        ][
            "overall"
        ][
            "AP"
        ]

        comparison[
            "methods"
        ][mid] = {
            "official_AP70":
                a70,
            "delta_official_AP70":
                a70 - b70,
            "route_AP50":
                a50,
            "delta_route_AP50":
                a50 - b50,
            "coco_AP":
                cap,
            "delta_coco_AP":
                cap - bcap,
        }

        rows.append({
            "experiment_id":
                mid,
            "adaptation_trainable_parameters":
                m[
                    "adaptation_trainable_parameters"
                ],
            "official_AP70_pct":
                a70 * 100.0,
            "delta_official_AP70_pct":
                (a70 - b70) * 100.0,
            "route_AP50_pct":
                a50 * 100.0,
            "delta_route_AP50_pct":
                (a50 - b50) * 100.0,
            "coco_AP":
                cap,
            "delta_coco_AP":
                cap - bcap,
            "coco_AP50":
                m[
                    "coco"
                ][
                    "overall"
                ][
                    "AP50"
                ],
            "coco_AP75":
                m[
                    "coco"
                ][
                    "overall"
                ][
                    "AP75"
                ],
            "coco_APsmall":
                m[
                    "coco"
                ][
                    "overall"
                ][
                    "APsmall"
                ],
            "coco_ARsmall":
                m[
                    "coco"
                ][
                    "overall"
                ][
                    "ARsmall"
                ],
            "inference_wall_seconds":
                m[
                    "inference_wall_seconds"
                ],
        })

    (
        output
        / "joint_summary.json"
    ).write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        output
        / "joint_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "G00_JOINT_EVALUATION=COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
