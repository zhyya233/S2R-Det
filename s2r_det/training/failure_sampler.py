from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


FAILURE_TYPES = ("F1", "F2", "F3", "F4")


def bounded_failure_quotas(
    counts: Mapping[str, int],
    lower: float = 0.10,
    upper: float = 0.40,
) -> Dict[str, float]:
    """Bounded proportional allocation.

    Start from effective failure-instance counts. Active failure types receive
    a quota in [lower, upper]. Fixed lower/upper allocations are removed from
    the remaining mass and the rest is redistributed proportionally.

    This is deterministic and does not use training results.
    """

    active = [
        k
        for k in FAILURE_TYPES
        if int(counts.get(k, 0)) > 0
    ]

    if not active:
        raise ValueError("no active F1-F4 types")

    if len(active) * lower > 1.0 + 1e-12:
        raise ValueError("lower bound infeasible")

    if len(active) * upper < 1.0 - 1e-12:
        raise ValueError("upper bound infeasible")

    free = set(active)
    fixed: Dict[str, float] = {}

    while free:
        remaining = 1.0 - sum(fixed.values())

        total = sum(
            float(counts[k])
            for k in free
        )

        if total <= 0:
            share = remaining / len(free)
            for k in free:
                fixed[k] = share
            free.clear()
            break

        proposal = {
            k: (
                remaining
                * float(counts[k])
                / total
            )
            for k in free
        }

        low = [
            k for k, v in proposal.items()
            if v < lower - 1e-15
        ]

        high = [
            k for k, v in proposal.items()
            if v > upper + 1e-15
        ]

        if not low and not high:
            fixed.update(proposal)
            free.clear()
            break

        # Freeze all violating bounds, then redistribute residual mass.
        for k in low:
            fixed[k] = lower
            free.remove(k)

        for k in high:
            if k in free:
                fixed[k] = upper
                free.remove(k)

    # Numerical normalization while preserving practical bounds.
    total = sum(fixed.values())

    if abs(total - 1.0) > 1e-12:
        # Put tiny floating residue into largest unconstrained-looking entry.
        k = max(
            fixed,
            key=lambda x: fixed[x],
        )
        fixed[k] += 1.0 - total

    for k, v in fixed.items():
        if v < lower - 1e-9 or v > upper + 1e-9:
            raise RuntimeError(
                f"bounded allocation failed: {k}={v}"
            )

    return {
        k: float(fixed[k])
        for k in FAILURE_TYPES
        if k in fixed
    }


@dataclass(frozen=True)
class FailureSamplerSpec:
    failure_batch_fraction: float
    retention_batch_fraction: float
    failure_quotas: Dict[str, float]
    batch_size: int
    seed: int


class FailureStratifiedBatchSource:
    """Deterministic source planner for fixed-step training.

    A complete batch is either:
      - failure batch: every image is drawn via an F1-F4 quota; or
      - retention batch: images are drawn uniformly from the frozen
        retention-image list.

    For a run of N optimizer steps, exactly round(0.70*N) batches are failure
    batches. The schedule is shuffled deterministically with the run seed.
    """

    def __init__(
        self,
        failure_pools: Mapping[str, Sequence[str]],
        retention_pool: Sequence[str],
        failure_quotas: Mapping[str, float],
        batch_size: int,
        seed: int,
        failure_batch_fraction: float = 0.70,
    ):
        self.failure_pools = {
            k: tuple(failure_pools[k])
            for k in FAILURE_TYPES
            if k in failure_pools
        }

        self.retention_pool = tuple(retention_pool)

        self.failure_quotas = dict(failure_quotas)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.failure_batch_fraction = float(failure_batch_fraction)

        if not 0.0 <= self.failure_batch_fraction <= 1.0:
            raise ValueError("failure_batch_fraction must be in [0, 1]")

        if not self.retention_pool:
            raise ValueError("empty retention pool")

        for k, pool in self.failure_pools.items():
            if not pool:
                raise ValueError(
                    f"empty failure pool: {k}"
                )

    def source_schedule(
        self,
        steps: int,
    ) -> List[str]:

        failure_steps = round(
            float(steps) * self.failure_batch_fraction
        )

        retention_steps = (
            int(steps) - failure_steps
        )

        schedule = (
            ["failure"] * failure_steps
            + ["retention"] * retention_steps
        )

        rng = random.Random(self.seed)
        rng.shuffle(schedule)

        return schedule

    def iter_batches(
        self,
        steps: int,
    ):
        rng = random.Random(self.seed)

        schedule = self.source_schedule(
            steps
        )

        types = list(
            self.failure_quotas
        )

        weights = [
            self.failure_quotas[k]
            for k in types
        ]

        for step, source in enumerate(schedule):
            if source == "retention":
                names = [
                    self.retention_pool[
                        rng.randrange(
                            len(self.retention_pool)
                        )
                    ]
                    for _ in range(self.batch_size)
                ]

                yield {
                    "step": step,
                    "source": source,
                    "failure_types": [],
                    "file_names": names,
                }

            else:
                chosen_types = rng.choices(
                    types,
                    weights=weights,
                    k=self.batch_size,
                )

                names = [
                    self.failure_pools[t][
                        rng.randrange(
                            len(self.failure_pools[t])
                        )
                    ]
                    for t in chosen_types
                ]

                yield {
                    "step": step,
                    "source": source,
                    "failure_types":
                        chosen_types,
                    "file_names": names,
                }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()
