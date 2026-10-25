"""IoU-priority GT diagnostics, distinct from score-ranked AP matching."""

import math


def box_iou(a, b):
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("four finite xyxy coordinates required")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("invalid box extent")
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def greedy_match(gt, predictions, threshold=0.5, class_consistent=True):
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold outside [0,1]")
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(predictions):
            if not class_consistent or g["category_id"] == p["category_id"]:
                overlap = box_iou(g["box"], p["box"])
                if overlap >= threshold:
                    pairs.append((-overlap, gi, pi))
    used = set()
    result = {}
    for _, gi, pi in sorted(pairs):
        if gi not in result and pi not in used:
            result[gi] = pi
            used.add(pi)
    return result


def transitions(gt, baseline, repaired):
    b = set(greedy_match(gt, baseline))
    r = set(greedy_match(gt, repaired))
    n = len(gt)
    recovered = len(r - b)
    damaged = len(b - r)
    return {
        "GT": n,
        "baseline_success": len(b),
        "recovered": recovered,
        "damaged": damaged,
        "net": recovered - damaged,
        "retained": len(b & r),
        "retained_keys": [str(gt[i]["annotation_id"]) for i in sorted(b & r)],
        "net_percent": 100 * (recovered - damaged) / n if n else None,
        "success_retention_percent": 100 * len(b & r) / len(b) if b else None,
        "recovered_keys": [str(gt[i]["annotation_id"]) for i in sorted(r - b)],
        "damaged_keys": [str(gt[i]["annotation_id"]) for i in sorted(b - r)],
    }
