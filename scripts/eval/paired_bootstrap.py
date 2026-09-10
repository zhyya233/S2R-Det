#!/usr/bin/env python3
import argparse
import copy
import gc
import io
import json
import os
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_EV = None
CAND_EV = None
BOOT = None


def prepare(gt_path, pred_path):
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(str(pred_path))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(gt.getImgIds())
        ev.params.catIds = sorted(gt.getCatIds())
        ev.params.areaRng = [[0.0, 32.0 * 32.0]]
        ev.params.areaRngLbl = ["small"]
        ev.params.maxDets = [100]
        ev.evaluate()
    return ev


def sample_ap(ev, sample):
    pe = ev._paramsEval
    n0 = len(pe.imgIds)
    a0 = len(pe.areaRng)
    k0 = len(pe.catIds)

    selected = []
    for k in range(k0):
        for a in range(a0):
            off = (k * a0 + a) * n0
            selected.extend(
                ev.evalImgs[off + int(i)]
                for i in sample
            )

    q = copy.copy(ev)
    q.params = copy.deepcopy(ev.params)
    q._paramsEval = copy.deepcopy(ev._paramsEval)

    synthetic = list(range(len(sample)))
    q.params.imgIds = synthetic
    q._paramsEval.imgIds = synthetic
    q.evalImgs = selected
    q.eval = {}
    q.accumulate()

    precision = q.eval["precision"]
    valid = precision[precision > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def worker(span):
    lo, hi = span
    out = np.empty(hi - lo, dtype=np.float64)

    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink):
            for j, r in enumerate(range(lo, hi)):
                idx = BOOT[r]
                out[j] = (
                    sample_ap(CAND_EV, idx)
                    - sample_ap(BASE_EV, idx)
                ) * 100.0

    return lo, out


def expected_ap(pred_path):
    p = Path(pred_path).with_name("metrics.json")
    d = json.loads(p.read_text())
    return float(d["coco"]["APsmall"])


def full_ap(ev):
    idx = np.arange(
        len(ev._paramsEval.imgIds),
        dtype=np.int32,
    )
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink):
            return sample_ap(ev, idx)


def main():
    global BASE_EV, CAND_EV, BOOT

    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("ID", "PREDICTIONS"),
        required=True,
    )
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260902)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    BASE_EV = prepare(a.gt, a.baseline)
    n = len(BASE_EV._paramsEval.imgIds)

    baseline_full = full_ap(BASE_EV)
    baseline_expected = expected_ap(a.baseline)

    if abs(baseline_full - baseline_expected) > 5e-12:
        raise RuntimeError(
            "baseline APsmall reproduction mismatch: "
            + repr((baseline_full, baseline_expected))
        )

    rng = np.random.default_rng(a.bootstrap_seed)
    BOOT = rng.integers(
        0,
        n,
        size=(a.iterations, n),
        dtype=np.int32,
    )

    summary = {
        "metric": "COCO APsmall",
        "unit": "AP point",
        "iterations": a.iterations,
        "bootstrap_seed": a.bootstrap_seed,
        "num_images": n,
        "baseline_APsmall": baseline_full,
        "baseline_reproduction_error":
            baseline_full - baseline_expected,
        "pairs": {},
    }
    saved = {}

    for cid, pred in a.candidate:
        CAND_EV = prepare(a.gt, pred)

        if (
            CAND_EV._paramsEval.imgIds
            != BASE_EV._paramsEval.imgIds
        ):
            raise RuntimeError(cid + ": image IDs differ")

        cand_full = full_ap(CAND_EV)
        cand_expected = expected_ap(pred)

        if abs(cand_full - cand_expected) > 5e-12:
            raise RuntimeError(
                cid + " APsmall reproduction mismatch: "
                + repr((cand_full, cand_expected))
            )

        deltas = np.empty(
            a.iterations,
            dtype=np.float64,
        )

        tasks = [
            (lo, min(lo + a.chunk, a.iterations))
            for lo in range(0, a.iterations, a.chunk)
        ]

        ctx = get_context("fork")
        workers = min(
            a.workers,
            os.cpu_count() or a.workers,
        )

        with ctx.Pool(processes=workers) as pool:
            for lo, vals in pool.imap_unordered(
                worker,
                tasks,
            ):
                deltas[lo:lo + len(vals)] = vals

        lo95, hi95 = np.quantile(
            deltas,
            [0.025, 0.975],
        )

        summary["pairs"][cid] = {
            "candidate_APsmall": cand_full,
            "candidate_reproduction_error":
                cand_full - cand_expected,
            "observed_delta_AP_points":
                (cand_full - baseline_full) * 100.0,
            "bootstrap_mean_delta_AP_points":
                float(deltas.mean()),
            "bootstrap_sd_AP_points":
                float(deltas.std(ddof=1)),
            "ci95_low_AP_points": float(lo95),
            "ci95_high_AP_points": float(hi95),
            "fraction_delta_positive":
                float(np.mean(deltas > 0.0)),
        }

        saved[cid.replace("-", "_")] = deltas
        CAND_EV = None
        gc.collect()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    np.savez_compressed(
        out.with_suffix(".npz"),
        **saved,
    )


if __name__ == "__main__":
    main()
