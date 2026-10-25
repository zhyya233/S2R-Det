"""Paper-derived mutually exclusive attribution with explicit NMS provenance."""

import math
from collections import Counter, defaultdict

from s2r_det.evaluation.matching import box_iou, greedy_match

FAILURES = ("F1", "F2", "F3", "F4")


def quantile(values, p):
    if not values or not 0 <= p <= 1 or not all(math.isfinite(v) for v in values):
        raise ValueError("finite nonempty quantile sample required")
    x = sorted(values)
    v = (len(x) - 1) * p
    k = math.floor(v)
    f = v - k
    return (1 - f) * x[k] + f * x[min(k + 1, len(x) - 1)]


def trace_nms(pre, iou_threshold, max_detections):
    if not 0 <= iou_threshold <= 1 or max_detections < 1:
        raise ValueError("invalid fixed inference policy")
    kept = []
    suppressed = []
    for i in sorted(range(len(pre)), key=lambda i: (-pre[i]["score"], i)):
        if any(
            pre[i]["category_id"] == pre[k]["category_id"]
            and box_iou(pre[i]["box"], pre[k]["box"]) > iou_threshold
            for k in kept
        ):
            suppressed.append(i)
        else:
            kept.append(i)
    return {
        "post_indices": kept[:max_detections],
        "suppressed_indices": suppressed,
        "capped_indices": kept[max_detections:],
    }


def scale_group(g):
    if g["input_area"] < 256:
        return "vt16"
    return "small_non_vt16" if g["area"] < 1024 else None


def validate_trace(pre, trace):
    keys = ("post_indices", "suppressed_indices", "capped_indices")
    if any(k not in trace for k in keys):
        raise ValueError("complete NMS/cap trace required")
    flat = [i for k in keys for i in trace[k]]
    if len(set(flat)) != len(flat) or set(flat) != set(range(len(pre))):
        raise ValueError("NMS trace must partition every candidate exactly once")


def preliminary(gt, pre, trace):
    if any(not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1 for p in pre):
        raise ValueError("candidate scores must be finite probabilities")
    if any(
        not math.isfinite(g["area"])
        or not math.isfinite(g["input_area"])
        or g["area"] <= 0
        or g["input_area"] <= 0
        for g in gt
    ):
        raise ValueError("GT areas must be finite and positive")
    validate_trace(pre, trace)
    post = [pre[i] for i in trace["post_indices"]]
    pm = greedy_match(gt, pre, 0.1)
    qm = greedy_match(gt, post, 0.1)
    out = []
    for gi, g in enumerate(gt):
        group = scale_group(g)
        if group is None:
            continue

        def best(candidates):
            return max(
                [
                    box_iou(g["box"], p["box"])
                    for p in candidates
                    if p["category_id"] == g["category_id"]
                ],
                default=0,
            )

        bp, bq = best(pre), best(post)
        sup = best([pre[i] for i in trace["suppressed_indices"]])
        pp = pre[pm[gi]] if gi in pm else None
        qp = post[qm[gi]] if gi in qm else None
        reason = ""
        score = None
        if sup >= 0.5 and bq < 0.5:
            if (
                gi in pm
                and pm[gi] in trace["suppressed_indices"]
                and box_iou(g["box"], pp["box"]) >= 0.5
            ):
                label = "F4"
            else:
                label = "ambiguous"
                reason = "one_to_one_suppression_conflict"
        elif max(bp, bq) < 0.1:
            label = "F1"
        elif max(bp, bq) < 0.5:
            if pp is not None or qp is not None:
                label = "F2"
            else:
                label = "ambiguous"
                reason = "one_to_one_localization_conflict"
        elif bq >= 0.5:
            if qp is not None and box_iou(g["box"], qp["box"]) >= 0.5:
                label = "success_pending"
                score = qp["score"]
            else:
                label = "ambiguous"
                reason = "one_to_one_success_conflict"
        else:
            label = "ambiguous"
            reason = "cap_or_unresolved_disappearance"
        out.append(
            {
                **g,
                "scale_group": group,
                "category": label,
                "score": score,
                "ambiguity_reason": reason,
            }
        )
    return out


def build_atlas(images, frozen_quartiles=None, fit_training_quartiles=False):
    if (frozen_quartiles is None) == (not fit_training_quartiles):
        raise ValueError("explicitly fit training quartiles OR supply frozen training quartiles")
    if frozen_quartiles is not None and any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in frozen_quartiles.values()
    ):
        raise ValueError("invalid frozen confidence quartile")
    records = []
    seen = set()
    for image in images:
        identity = str(image["image_id"])
        if identity in seen:
            raise ValueError("duplicate image identity; include sequence namespace")
        seen.add(identity)
        rows = preliminary(image["gt"], image["pre"], image["trace"])
        for r in rows:
            r["image_id"] = identity
        records.extend(rows)
    if fit_training_quartiles:
        groups = defaultdict(list)
        for r in records:
            if r["category"] == "success_pending":
                groups[f"{r['category_id']}:{r['scale_group']}"].append(r["score"])
        frozen_quartiles = {k: quantile(v, 0.25) for k, v in groups.items()}
    for r in records:
        if r["category"] == "success_pending":
            key = f"{r['category_id']}:{r['scale_group']}"
            if key not in frozen_quartiles:
                r["category"] = "ambiguous"
                r["ambiguity_reason"] = "missing_frozen_training_quartile"
            else:
                r["category"] = "F3" if r["score"] <= frozen_quartiles[key] else "F5"
    pools, counts, retention = image_pools(records)
    return {
        "records": records,
        "quartiles": frozen_quartiles,
        "failure_pools": pools,
        "failure_type_counts": counts,
        "retention_pool": retention,
        "category_counts": dict(Counter(r["category"] for r in records)),
        "association_population": "all retained GT before scope filtering",
    }


def image_pools(records):
    by_image = defaultdict(Counter)
    pools = {k: set() for k in FAILURES}
    counts = Counter()
    seen = set()
    for r in records:
        key = (str(r["image_id"]), str(r["annotation_id"]))
        if key in seen:
            raise ValueError("duplicate GT identity")
        seen.add(key)
        c = r["category"]
        name = str(r["image_id"])
        by_image[name][c] += 1
        if c in FAILURES:
            pools[c].add(name)
            counts[c] += 1
    retention = sorted(
        n for n, c in by_image.items() if c["F5"] > 0 and c["F5"] > sum(c[k] for k in FAILURES)
    )
    return {k: sorted(v) for k, v in pools.items()}, dict(counts), retention
