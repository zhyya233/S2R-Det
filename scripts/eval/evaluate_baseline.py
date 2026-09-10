import argparse
import copy
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def py(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if hasattr(v, "__float__"):
        return float(v)
    return v


def formal_mmdet_val(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = ckpt
    cfg.work_dir = "/tmp/s2r_b00_formal_val"

    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.test_cfg = dict(type="TestLoop")

    runner = Runner.from_cfg(cfg)
    out = runner.test()
    return {k: py(v) for k, v in out.items()}


def load_raw_gt(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 8), dtype=np.int32)

    x = np.loadtxt(
        p, delimiter=",", dtype=np.int32,
        ndmin=2, usecols=range(8)
    )
    return x.reshape(-1, 8)


def coco_metrics(gt_path, dets):
    gt = COCO(gt_path)
    dt = gt.loadRes(dets) if dets else gt.loadRes([])

    ev = COCOeval(gt, dt, "bbox")
    ev.params.maxDets = [1, 10, 500]
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
        "AR500": float(s[8]),
        "ARsmall": float(s[9]),
        "ARmedium": float(s[10]),
        "ARlarge": float(s[11]),
    }


def vt16_transform(dataset, dets):
    ds = copy.deepcopy(dataset)
    scales = {}

    for im in ds["images"]:
        s = min(
            640.0 / float(im["width"]),
            640.0 / float(im["height"])
        )
        scales[int(im["id"])] = s
        im["width"] = 640
        im["height"] = 640

    for a in ds["annotations"]:
        s = scales[int(a["image_id"])]
        x, y, w, h = map(float, a["bbox"])
        a["bbox"] = [x*s, y*s, w*s, h*s]
        a["area"] = w*h*s*s

    out = []
    for d in dets:
        s = scales[int(d["image_id"])]
        x, y, w, h = map(float, d["bbox"])
        q = dict(d)
        q["bbox"] = [x*s, y*s, w*s, h*s]
        out.append(q)

    return ds, out


def vt16_eval(dataset, dets):
    ds, dt_rows = vt16_transform(dataset, dets)

    gt = COCO()
    gt.dataset = ds
    gt.createIndex()
    dt = gt.loadRes(dt_rows) if dt_rows else gt.loadRes([])

    ev = COCOeval(gt, dt, "bbox")
    ev.params.areaRng = [[0.0, 256.0]]
    ev.params.areaRngLbl = ["vt16"]
    ev.params.maxDets = [1, 10, 500]

    ev.evaluate()
    ev.accumulate()

    p = ev.eval["precision"][:, :, :, 0, 2]
    p = p[p > -1]

    r = ev.eval["recall"][:, :, 0, 2]
    r = r[r > -1]

    r50 = ev.eval["recall"][0, :, 0, 2]
    r50 = r50[r50 > -1]

    n = sum(
        float(a["area"]) < 256.0
        for a in ds["annotations"]
    )

    return {
        "num_gt": int(n),
        "AP": float(np.mean(p)) if p.size else float("nan"),
        "Recall": float(np.mean(r)) if r.size else float("nan"),
        "Recall50": float(np.mean(r50)) if r50.size else float("nan"),
        "definition":
            "GT area after keep-ratio 640 transform < 16^2; "
            "IoU=.50:.05:.95; maxDets=500"
    }


def vt16_unit():
    ds = {
        "info": {},
        "licenses": [],
        "images": [{
            "id": 1,
            "file_name": "x.jpg",
            "width": 640,
            "height": 640
        }],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 10, 10, 10],
                "area": 100, "iscrowd": 0
            },
            {
                "id": 2, "image_id": 1, "category_id": 1,
                "bbox": [100, 100, 30, 30],
                "area": 900, "iscrowd": 0
            }
        ]
    }

    det = [
        {
            "image_id": 1, "category_id": 1,
            "bbox": [10, 10, 10, 10], "score": .99
        },
        {
            "image_id": 1, "category_id": 1,
            "bbox": [100, 100, 30, 30], "score": .98
        }
    ]

    x = vt16_eval(ds, det)
    ok = (
        x["num_gt"] == 1
        and math.isfinite(x["AP"])
        and math.isfinite(x["Recall"])
        and abs(x["AP"] - 1.0) < 1e-8
        and abs(x["Recall"] - 1.0) < 1e-8
    )
    return "PASS" if ok else "FAIL", x


def official_visdrone(toolkit, gt_rows, det_rows, hs, ws):
    sys.path.insert(0, toolkit)
    from viseval import eval_det

    sig = inspect.signature(eval_det)
    kw = {}
    if "per_class" in sig.parameters:
        kw["per_class"] = True

    raw = eval_det(gt_rows, det_rows, hs, ws, **kw)

    if not isinstance(raw, (tuple, list)):
        raise RuntimeError(
            "Unexpected eval_det return: " + repr(type(raw))
        )

    vals = list(raw)
    if len(vals) < 7:
        raise RuntimeError(
            "Unexpected eval_det output length: " + str(len(vals))
        )

    out = {
        "AP_pct": float(vals[0]),
        "AP50_pct": float(vals[1]),
        "AP75_pct": float(vals[2]),
        "AR1_pct": float(vals[3]),
        "AR10_pct": float(vals[4]),
        "AR100_pct": float(vals[5]),
        "AR500_pct": float(vals[6]),
    }

    if len(vals) >= 8:
        try:
            out["classwise_AP_pct"] = [
                float(v) for v in np.asarray(vals[7]).reshape(-1)
            ]
        except Exception:
            out["classwise_AP_pct"] = []

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coco", required=True)
    p.add_argument("--val-root", required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    vt_status, vt_unit_detail = vt16_unit()
    if vt_status != "PASS":
        raise RuntimeError(
            "VT16 synthetic unit test failed: "
            + repr(vt_unit_detail)
        )

    formal = formal_mmdet_val(a.config, a.checkpoint)

    with open(a.coco, encoding="utf-8") as f:
        ds = json.load(f)

    images = sorted(ds["images"], key=lambda z: int(z["id"]))

    model = init_detector(
        a.config, a.checkpoint, device="cuda:0"
    )

    dets = []
    raw_gt = []
    raw_dt = []
    hs, ws = [], []

    bs = 16
    for st in range(0, len(images), bs):
        batch = images[st:st+bs]
        paths = [
            str(Path(a.val_root) / "images" / im["file_name"])
            for im in batch
        ]

        preds = inference_detector(model, paths)
        if not isinstance(preds, list):
            preds = [preds]

        for im, res in zip(batch, preds):
            iid = int(im["id"])
            stem = Path(im["file_name"]).stem
            hs.append(int(im["height"]))
            ws.append(int(im["width"]))

            gt = load_raw_gt(
                Path(a.val_root) / "annotations" /
                (stem + ".txt")
            )
            raw_gt.append(gt)

            ins = res.pred_instances.cpu()
            boxes = ins.bboxes.numpy()
            scores = ins.scores.numpy()
            labels = ins.labels.numpy()

            order = np.argsort(-scores)
            rows = []

            for j in order:
                x1, y1, x2, y2 = map(float, boxes[j])
                w = max(0.0, x2-x1)
                h = max(0.0, y2-y1)
                score = float(scores[j])
                cat = int(labels[j]) + 1

                rows.append(
                    [x1, y1, w, h, score, cat, -1, -1]
                )
                dets.append({
                    "image_id": iid,
                    "category_id": cat,
                    "bbox": [x1, y1, w, h],
                    "score": score
                })

            raw_dt.append(
                np.asarray(rows, dtype=np.float64)
                if rows else
                np.zeros((0, 8), dtype=np.float64)
            )

    official = official_visdrone(
        a.toolkit, raw_gt, raw_dt, hs, ws
    )
    coco = coco_metrics(a.coco, dets)
    vt16 = vt16_eval(ds, dets)

    result = {
        "vt16_unit_test": vt_status,
        "formal_mmdet": formal,
        "official_visdrone": official,
        "coco_diagnostic": coco,
        "vt16": vt16,
        "num_images": len(images),
        "prediction_count": len(dets),
        "model": {
            "total_parameters":
                int(sum(x.numel() for x in model.parameters())),
            "trainable_parameters":
                int(sum(
                    x.numel() for x in model.parameters()
                    if x.requires_grad
                ))
        }
    }

    Path(a.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
