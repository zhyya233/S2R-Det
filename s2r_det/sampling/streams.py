"""Deterministic update streams with full-image supervision."""

import math
import random
from s2r_det.failure_atlas.atlas import FAILURES


def bounded_failure_quotas(counts, lower=0.1, upper=0.4):
    if any(not isinstance(v, int) or v < 0 for v in counts.values()):
        raise ValueError("counts must be nonnegative integers")
    active = [k for k in FAILURES if counts.get(k, 0) > 0]
    if (
        not active
        or lower < 0
        or upper < lower
        or len(active) * lower > 1
        or len(active) * upper < 1
    ):
        raise ValueError("active failure quota bounds infeasible")
    lo = 0.0
    hi = 1.0
    total = lambda a: sum(min(upper, max(lower, a * counts[k])) for k in active)
    while total(hi) < 1:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if total(mid) < 1:
            lo = mid
        else:
            hi = mid
    out = {k: min(upper, max(lower, (lo + hi) / 2 * counts[k])) for k in active}
    if abs(sum(out.values()) - 1) > 1e-10:
        raise RuntimeError("bounded allocation failed")
    return out


class FailureStratifiedBatchSource:
    def __init__(
        self,
        failure_pools,
        retention_pool,
        failure_quotas,
        batch_size=16,
        seed=0,
        failure_batch_fraction=0.7,
    ):
        if batch_size < 1 or int(batch_size) != batch_size or not 0 <= failure_batch_fraction <= 1:
            raise ValueError("invalid stream settings")
        self.failure_pools = {k: tuple(sorted(set(v))) for k, v in failure_pools.items() if v}
        self.retention_pool = tuple(sorted(set(retention_pool)))
        self.quotas = dict(failure_quotas)
        if failure_batch_fraction > 0 and (
            not self.quotas or any(k not in self.failure_pools for k in self.quotas)
        ):
            raise ValueError("required failure pool empty")
        if failure_batch_fraction < 1 and not self.retention_pool:
            raise ValueError("required retention pool empty")
        if self.quotas and (
            any(v <= 0 or not math.isfinite(v) for v in self.quotas.values())
            or abs(sum(self.quotas.values()) - 1) > 1e-9
        ):
            raise ValueError("invalid quotas")
        self.batch_size = batch_size
        self.seed = seed
        self.fraction = failure_batch_fraction

    def source_schedule(self, steps):
        if steps < 1 or int(steps) != steps:
            raise ValueError("positive integer steps required")
        n = math.floor(self.fraction * steps + 0.5)
        schedule = ["failure"] * n + ["retention"] * (steps - n)
        random.Random(self.seed).shuffle(schedule)
        return schedule

    def iter_batches(self, steps):
        rng = random.Random(self.seed)
        types = sorted(self.quotas)
        weights = [self.quotas[k] for k in types]
        for step, source in enumerate(self.source_schedule(steps)):
            chosen = (
                rng.choices(types, weights=weights, k=self.batch_size)
                if source == "failure"
                else []
            )
            names = (
                [rng.choice(self.failure_pools[k]) for k in chosen]
                if chosen
                else [rng.choice(self.retention_pool) for _ in range(self.batch_size)]
            )
            yield {"step": step, "source": source, "image_ids": names, "failure_types": chosen}


def uniform_batches(image_ids, steps, batch_size=16, seed=0):
    ids = sorted(set(image_ids))
    if not ids or steps < 1 or batch_size < 1:
        raise ValueError("empty support or invalid budget")
    rng = random.Random(seed)
    for step in range(steps):
        yield {"step": step, "source": "uniform", "image_ids": rng.choices(ids, k=batch_size)}


def subset_support(image_ids, count, seed=0, scores=None):
    ids = sorted(set(image_ids))
    if not 0 < count <= len(ids):
        raise ValueError("explicit subset count outside population")
    if scores is None:
        return sorted(random.Random(seed).sample(ids, count))
    if set(scores) != set(ids) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("provide one finite fixed loss score per image")
    return sorted(ids, key=lambda k: (-scores[k], k))[:count]
