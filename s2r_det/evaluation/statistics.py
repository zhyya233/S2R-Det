"""Training-seed summaries and paired resampling have separate interfaces."""

import math
import random
from statistics import mean, stdev
from s2r_det.failure_atlas.atlas import quantile


def seed_summary(values):
    values = list(values)
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("sample SD requires at least two finite independent seed values")
    return {
        "mean": mean(values),
        "sample_SD": stdev(values),
        "seeds": len(values),
        "statistical_unit": "independently trained repair seed",
    }


def paired_bootstrap(units, baseline_metric, repair_metric, replicates=10000, seed=0, unit="image"):
    if not units or len(set(units)) != len(units) or replicates < 1:
        raise ValueError("unique paired sampling units and positive replicate count required")
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choices(units, k=len(units))
        value = repair_metric(sampled) - baseline_metric(sampled)
        if not math.isfinite(value):
            raise ValueError("undefined bootstrap endpoint; no replicate silently omitted")
        draws.append(value)
    point = repair_metric(units) - baseline_metric(units)
    return {
        "point_difference": point,
        "CI95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": unit,
        "interval": "paired percentile; not a training-seed SD",
    }


def paired_sequence_bootstrap(baseline_by_sequence, repair_by_sequence, replicates=10000, seed=0):
    if set(baseline_by_sequence) != set(repair_by_sequence):
        raise ValueError("sequence identities differ")
    units = sorted(baseline_by_sequence)
    return paired_bootstrap(
        units,
        lambda ids: mean(baseline_by_sequence[i] for i in ids),
        lambda ids: mean(repair_by_sequence[i] for i in ids),
        replicates,
        seed,
        "sequence; equal-weight sequence-mean AP70",
    )


def replicate_coco(dataset, predictions, image_ids):
    """Each repeated image is an independent COCO image/annotation replica."""
    import copy

    images = {str(i["id"]): i for i in dataset["images"]}
    out = copy.deepcopy(dataset)
    out["images"] = []
    out["annotations"] = []
    dt = []
    for index, identity in enumerate(image_ids):
        identity = str(identity)
        if identity not in images:
            raise ValueError("unknown resampled image")
        out["images"].append({**images[identity], "id": index + 1})
        for a in dataset["annotations"]:
            if str(a["image_id"]) == identity:
                out["annotations"].append(
                    {**a, "id": len(out["annotations"]) + 1, "image_id": index + 1}
                )
        dt.extend(
            {**p, "image_id": index + 1} for p in predictions if str(p["image_id"]) == identity
        )
    return out, dt
