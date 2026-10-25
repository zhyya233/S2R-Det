"""Semantic-group log medians; preserve every instance in resampled images."""

import itertools
import math
import random
from collections import defaultdict
from statistics import median
from s2r_det.failure_atlas.atlas import FAILURES, quantile

GROUPS = ("backbone_shallow", "backbone_late", "neck_p3", "neck_p4", "head")
EPS = 1e-12


def group_logs(module_sensitivity, manifest):
    out = {}
    for group in GROUPS:
        modules = [m["path"] for m in manifest if m["group"] == group and m["analysis_eligible"]]
        if not modules:
            continue
        vals = [module_sensitivity[n] for n in modules]
        if any(not math.isfinite(x) or x < 0 for x in vals):
            raise ValueError("non-finite/negative sensitivity invalidates analysis")
        out[group] = median([math.log(v + EPS) for v in vals])
    return out


def contrasts(records, groups):
    out = {}
    for g in groups:
        f = [r["h"][g] for r in records if r["category"] in FAILURES]
        s = [r["h"][g] for r in records if r["category"] == "F5"]
        if not f or not s:
            raise ValueError("empty failure or retained-success instance population")
        if not all(math.isfinite(v) for v in f + s):
            raise ValueError("non-finite log score")
        out[g] = median(f) - median(s)
    return out


def bootstrap_stability(records, cohorts, replicates=1000, seed=0):
    if replicates < 1 or not records:
        raise ValueError("nonempty records and positive replicate count required")
    groups = [g for g in GROUPS if g in records[0]["h"]]
    if any(set(r["h"]) != set(groups) for r in records):
        raise ValueError("inconsistent eligible group set")
    by_image = defaultdict(list)
    cells = defaultdict(list)
    for r in records:
        if str(r["image_id"]) not in cohorts:
            raise ValueError("score outside fixed cohort")
        if r["category"] not in (*FAILURES, "F5"):
            raise ValueError("ambiguous/out-of-scope score supplied")
        by_image[str(r["image_id"])].append(r)
    for image, meta in sorted(cohorts.items()):
        if image not in by_image:
            raise ValueError("cohort image has no eligible instance records")
        cells[(meta["cohort"], meta["density"])].append(image)
    observed = contrasts(records, groups)
    samples = {g: [] for g in groups}
    top1 = defaultdict(int)
    top2 = defaultdict(int)
    rng = random.Random(seed)
    for _ in range(replicates):
        batch = []
        for _, images in sorted(cells.items()):
            for image in rng.choices(images, k=len(images)):
                batch.extend(by_image[image])
        q = contrasts(batch, groups)
        ranked = sorted(groups, key=lambda g: (-q[g], GROUPS.index(g)))
        for g in groups:
            samples[g].append(q[g])
        top1[ranked[0]] += 1
        for g in ranked[:2]:
            top2[g] += 1
    return {
        "groups": {
            g: {
                "q": observed[g],
                "CI95": [quantile(samples[g], 0.025), quantile(samples[g], 0.975)],
                "top1_frequency": top1[g] / replicates,
                "top2_frequency": top2[g] / replicates,
            }
            for g in groups
        },
        "replicates": replicates,
        "seed": seed,
        "unit": "image within cohort-by-density stratum; all F/F5 instances retained",
    }


def select_placement(result, manifest, budget, rank=8, number_groups=1, threshold=0.6):
    if number_groups not in (1, 2) or rank < 1 or budget < 1:
        raise ValueError("invalid placement budget")
    stable = [
        g
        for g in GROUPS
        if g in result["groups"]
        and result["groups"][g][f"top{number_groups}_frequency"] >= threshold
    ]
    ranked = sorted(stable, key=lambda g: (-result["groups"][g]["q"], GROUPS.index(g)))
    for combination in itertools.combinations(ranked, number_groups):
        sites = [m for m in manifest if m["group"] in combination and m["insertion_eligible"]]
        if any(not any(m["group"] == g for m in sites) for g in combination):
            continue
        cost = sum(rank * (m["in_channels"] + m["out_channels"] + 9) for m in sites)
        if cost <= budget:
            return {
                "groups": list(combination),
                "paths": [m["path"] for m in sites],
                "rank": rank,
                "trainable_parameters": cost,
            }
    raise ValueError(
        "automatic placement unavailable: no stable feasible insertion set; no manual fallback"
    )
