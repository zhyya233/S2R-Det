"""COCO-native area endpoints and a strict input-area VT16 endpoint."""

import copy
import contextlib
import io
import math


def evaluate_coco(
    dataset, predictions, max_detections=100, iou_thresholds=None, use_categories=True, vt16=False
):
    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("COCO evaluation requires numpy and pycocotools") from exc
    ds = copy.deepcopy(dataset)
    dt = copy.deepcopy(predictions)
    if vt16:
        from s2r_det.data.coco import resize_geometry

        scales = {i["id"]: resize_geometry(i["width"], i["height"]) for i in ds["images"]}
        for obj in ds["annotations"] + dt:
            _, _, sx, sy = scales[obj["image_id"]]
            x, y, w, h = obj["bbox"]
            obj["bbox"] = [sx * x, sy * y, sx * w, sy * h]
            obj["area"] = sx * sy * w * h
    ds.setdefault("info", {})
    ds.setdefault("licenses", [])
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = ds
        gt.createIndex()
        if dt:
            pred = gt.loadRes(dt)
        else:
            pred = COCO()
            pred.dataset = {**ds, "annotations": []}
            pred.createIndex()
        ev = COCOeval(gt, pred, "bbox")
        ev.params.maxDets = [1, 10, max_detections]
        ev.params.useCats = int(use_categories)
        if iou_thresholds is not None:
            ev.params.iouThrs = np.asarray(iou_thresholds)
        if vt16:
            ev.params.areaRng = [[0.0, math.nextafter(256.0, 0.0)]]
            ev.params.areaRngLbl = ["vt16"]
        ev.evaluate()
        ev.accumulate()

    def finite_mean(a):
        a = a[a >= 0]
        return float(a.mean() * 100) if a.size else None

    areas = ev.params.areaRngLbl
    out = {
        "AP": finite_mean(ev.eval["precision"][:, :, :, 0, -1]),
        "metric_unit": "percentage points",
        "max_detections": max_detections,
        "evaluator": "pycocotools COCOeval; 101 recall thresholds",
        "class_aware": use_categories,
        "area_endpoint": (
            "strict input area<256" if vt16 else "COCO original-image area, inclusive endpoints"
        ),
    }
    if not vt16:
        for name in ["small", "medium", "large"]:
            i = areas.index(name)
            out["AP" + name] = finite_mean(ev.eval["precision"][:, :, :, i, -1])
            out["AR" + name] = finite_mean(ev.eval["recall"][:, :, i, -1])
    if iou_thresholds is not None:
        for i, t in enumerate(iou_thresholds):
            out[f"AP{round(t*100)}"] = finite_mean(ev.eval["precision"][i, :, :, 0, -1])
    return out


def standard_metrics(dataset, predictions):
    standard = evaluate_coco(dataset, predictions, 100)
    ar = evaluate_coco(dataset, predictions, 500)
    vt = evaluate_coco(dataset, predictions, 500, vt16=True)
    return {
        **{k: standard[k] for k in ["AP", "APsmall", "APmedium", "APlarge"]},
        "ARsmall@500": ar["ARsmall"],
        "VT16 AP@500": vt["AP"],
        "metric_unit": "percentage points",
        "computed_from_supplied_predictions": True,
    }


def uavdt_metrics(dataset, predictions, max_detections):
    class_aware = evaluate_coco(dataset, predictions, 100)
    pooled = evaluate_coco(dataset, predictions, max_detections, [0.5, 0.7], False)
    seqs = sorted(
        {i.get("sequence_id") for i in dataset["images"] if i.get("sequence_id") is not None}
    )
    if not seqs or any(i.get("sequence_id") is None for i in dataset["images"]):
        raise ValueError("every UAVDT frame requires sequence_id")
    sequence_ap = {}
    for seq in seqs:
        ims = [i for i in dataset["images"] if i["sequence_id"] == seq]
        ids = {i["id"] for i in ims}
        subset = {
            **dataset,
            "images": ims,
            "annotations": [a for a in dataset["annotations"] if a["image_id"] in ids],
        }
        sequence_ap[seq] = evaluate_coco(
            subset, [p for p in predictions if p["image_id"] in ids], max_detections, [0.7], False
        )["AP70"]
    if any(v is None for v in sequence_ap.values()):
        raise ValueError("undefined sequence AP; sequence cannot be silently omitted")
    return {
        "class_aware_AP": class_aware["AP"],
        "pooled_class_agnostic_AP50": pooled["AP50"],
        "pooled_class_agnostic_AP70": pooled["AP70"],
        "sequence_AP70": sequence_ap,
        "equal_weight_sequence_AP70": sum(sequence_ap.values()) / len(sequence_ap),
        "class_agnostic_evaluator": "COCO 101-point implementation; historical evaluator equivalence unverified",
    }
