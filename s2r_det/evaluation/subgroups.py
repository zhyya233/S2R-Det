"""Separate scale/density/occlusion/VT16 partitions; never sum their totals."""

from collections import defaultdict
from s2r_det.evaluation.matching import greedy_match
from s2r_det.sensitivity.cohorts import density_label


def subgroup_transitions(images, training_density_thresholds):
    out = defaultdict(lambda: {"GT": 0, "recovered": 0, "damaged": 0, "baseline_success": 0})
    missing_occlusion = 0
    for image in images:
        gt = image["gt"]
        baseline = set(greedy_match(gt, image["baseline"]))
        repaired = set(greedy_match(gt, image["repaired"]))
        density = density_label(len(gt), training_density_thresholds)
        for i, g in enumerate(gt):
            # Exclusive subgroup counts assign boundary ties to the lower size group.
            scale = "small" if g["area"] <= 1024 else "medium" if g["area"] <= 4096 else "large"
            labels = [
                ("scale", scale),
                ("density", density),
                ("input_size", "vt16" if g["input_area"] < 256 else "non_vt16"),
            ]
            if g.get("occlusion") in (0, 1, 2):
                labels.append(("occlusion", str(g["occlusion"])))
            else:
                missing_occlusion += 1
            for dimension, label in labels:
                item = out[f"{dimension}:{label}"]
                item["GT"] += 1
                item["baseline_success"] += int(i in baseline)
                item["recovered"] += int(i in repaired and i not in baseline)
                item["damaged"] += int(i in baseline and i not in repaired)
    for item in out.values():
        item["net"] = item["recovered"] - item["damaged"]
        item["net_percent"] = 100 * item["net"] / item["GT"]
    return {
        "groups": dict(out),
        "missing_occlusion_GT": missing_occlusion,
        "scale_boundary_policy": "exclusive lower-group tie allocation for partition summaries; COCO AP itself uses native inclusive overlapping endpoints",
    }
