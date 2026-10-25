"""Training-only density thresholds and fixed disjoint analysis cohorts."""

import math
import random
from collections import Counter, defaultdict
from s2r_det.failure_atlas.atlas import FAILURES, quantile

DENSITIES = ("low", "medium", "high")


def density_thresholds(image_gt_counts):
    return [quantile(list(image_gt_counts.values()), p) for p in (1 / 3, 2 / 3)]


def density_label(n, thresholds):
    return "low" if n <= thresholds[0] else "medium" if n <= thresholds[1] else "high"


def build_cohorts(records, image_gt_counts, per_cohort=256, seed=0):
    if per_cohort < 1:
        raise ValueError("positive cohort size required")
    boundaries = density_thresholds(image_gt_counts)
    counts = defaultdict(Counter)
    for r in records:
        if str(r["image_id"]) not in image_gt_counts:
            raise ValueError("Atlas image absent from training density population")
        counts[str(r["image_id"])][r["category"]] += 1
    cells = defaultdict(list)
    for image, c in counts.items():
        f = sum(c[k] for k in FAILURES)
        s = c["F5"]
        cohort = "failure_rich" if f > 0 and f >= s else ("success_rich" if s > f else None)
        if cohort:
            cells[(cohort, density_label(image_gt_counts[image], boundaries))].append(image)
    rng = random.Random(seed)
    chosen = {}
    for cohort in ("failure_rich", "success_rich"):
        sizes = [len(cells[(cohort, d)]) for d in DENSITIES]
        total = sum(sizes)
        if total < per_cohort:
            raise ValueError(f"insufficient {cohort} images: {total} < {per_cohort}")
        exact = [per_cohort * n / total for n in sizes]
        allocation = [math.floor(v) for v in exact]
        order = sorted(range(3), key=lambda i: (-(exact[i] - allocation[i]), i))
        for i in order[: per_cohort - sum(allocation)]:
            allocation[i] += 1
        for density, n in zip(DENSITIES, allocation):
            for image in rng.sample(sorted(cells[(cohort, density)]), n):
                chosen[image] = {"cohort": cohort, "density": density}
    return {
        "images": chosen,
        "density_thresholds": boundaries,
        "seed": seed,
        "per_cohort": per_cohort,
    }
