"""Descriptive public commands; optional runtime imports occur after parsing."""

import argparse
from s2r_det.utils.io import load_json, save_json


def parser():
    p = argparse.ArgumentParser(
        prog="s2r-det",
        description="Paper-derived supervised detector repair; external data and local weights required.",
    )
    commands = p.add_subparsers(dest="command", required=True)
    for name in ["infer", "manifest", "sensitivity", "cache", "train"]:
        q = commands.add_parser(name)
        for flag in ["detector-config", "checkpoint", "annotations", "image-root"]:
            q.add_argument("--" + flag, required=True)
        q.add_argument("--device", default="cpu")
        q.add_argument("--adapter-factory")
        q.add_argument("--output", required=True)
        if name in {"sensitivity", "cache"}:
            q.add_argument("--atlas", required=True)
        if name == "manifest":
            q.add_argument("--insertion-sets", required=True)
            q.add_argument("--group-overrides")
        if name == "sensitivity":
            q.add_argument("--manifest", required=True)
            q.add_argument("--budget", required=True, type=int)
            q.add_argument("--rank", type=int, default=8)
            q.add_argument("--groups", type=int, choices=[1, 2], default=1)
        if name == "train":
            q.add_argument("--recipe", required=True)
            q.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
            q.add_argument(
                "--learning-rate",
                type=float,
                help="explicit override; required when the recipe leaves learning_rate unset",
            )
            for flag in [
                "atlas",
                "manifest",
                "selection",
                "placement",
                "retention-cache",
                "support",
            ]:
                q.add_argument("--" + flag)
    q = commands.add_parser("atlas")
    q.add_argument("--predictions", required=True)
    q.add_argument("--split", choices=["train", "validation", "test-dev"], required=True)
    q.add_argument("--fit-training-quartiles", action="store_true")
    q.add_argument("--quartiles")
    q.add_argument("--output", required=True)
    q = commands.add_parser("evaluate")
    q.add_argument("--annotations", required=True)
    q.add_argument("--predictions", required=True)
    q.add_argument("--dataset", choices=["visdrone", "uavdt"], default="visdrone")
    q.add_argument("--max-detections", type=int, default=500)
    q.add_argument("--output", required=True)
    q = commands.add_parser("diagnostics")
    q.add_argument("--baseline", required=True)
    q.add_argument("--repaired", required=True)
    q.add_argument("--output", required=True)
    q = commands.add_parser("bootstrap")
    q.add_argument("--unit", choices=["image", "sequence"], required=True)
    q.add_argument("--baseline", required=True)
    q.add_argument("--repaired", required=True)
    q.add_argument("--annotations")
    q.add_argument("--replicates", type=int, default=10000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--output", required=True)
    q = commands.add_parser("aggregate-seeds")
    q.add_argument("--records", required=True)
    q.add_argument("--output", required=True)
    q = commands.add_parser("convert-visdrone")
    q.add_argument("--split-root", required=True)
    q.add_argument("--output", required=True)
    q = commands.add_parser("prepare-uavdt")
    q.add_argument("--annotations", required=True)
    q.add_argument("--output", required=True)
    q = commands.add_parser("deployment")
    q.add_argument(
        "--factory",
        required=True,
        help="module:function returning decoded images, callback, optional synchronization, and an explicitly measured profile",
    )
    q.add_argument("--samples", type=int, default=1000)
    q.add_argument("--warmup", type=int, default=20)
    q.add_argument("--output", required=True)
    q = commands.add_parser("sampling-support")
    q.add_argument("--image-ids", required=True)
    q.add_argument("--count", required=True, type=int)
    q.add_argument("--kind", choices=["random_subset", "top_loss"], required=True)
    q.add_argument("--scores")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--output", required=True)
    q = commands.add_parser("degradation")
    q.add_argument(
        "--factory",
        required=True,
        help="module:function returning dataset, image_loader and predict callback; no checkpoint distributed",
    )
    q.add_argument("--seed", type=int, choices=[0], default=0)
    q.add_argument("--output", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "sampling-support":
        from s2r_det.sampling.streams import subset_support
        from s2r_det.utils.io import file_sha256

        if a.kind == "top_loss" and not a.scores:
            raise ValueError("Top-loss requires externally computed fixed image-loss scores")
        scores = load_json(a.scores) if a.scores else None
        return save_json(
            a.output,
            {
                "sampling_kind": a.kind,
                "image_ids": subset_support(load_json(a.image_ids), a.count, a.seed, scores),
                "seed": a.seed,
                "score_source_identity": file_sha256(a.scores) if a.scores else None,
                "selection_rule": (
                    "descending supplied score, image-ID tie break"
                    if scores is not None
                    else "seeded without-replacement support selection"
                ),
                "historical_equivalence": "UNVERIFIED; count/support-construction inputs explicitly supplied",
            },
        )
    if a.command == "degradation":
        import importlib
        from s2r_det.evaluation.degradations import evaluate_profiles

        module, name = a.factory.split(":", 1)
        bundle = getattr(importlib.import_module(module), name)()
        return save_json(
            a.output,
            evaluate_profiles(bundle["dataset"], bundle["image_loader"], bundle["predict"], a.seed),
        )
    if a.command in [
        "infer",
        "atlas",
        "manifest",
        "sensitivity",
        "cache",
        "train",
        "evaluate",
        "diagnostics",
    ]:
        from s2r_det import workflows

        return getattr(workflows, a.command)(a)
    if a.command == "convert-visdrone":
        from s2r_det.data.coco import convert_visdrone

        return save_json(a.output, convert_visdrone(a.split_root))
    if a.command == "prepare-uavdt":
        ds = load_json(a.annotations)
        ids = [i["id"] for i in ds["images"]]
        if len({i["file_name"] for i in ds["images"]}) != len(ids):
            raise ValueError(
                "include sequence namespace in file_name; repeated frame basenames are unsafe"
            )
        if len(ids) != len(set(ids)) or any(not i.get("sequence_id") for i in ds["images"]):
            raise ValueError(
                "supply converted UAVDT COCO with globally unique frame IDs and explicit sequence_id; original split/ignore policy must be externally supplied"
            )
        return save_json(a.output, ds)
    if a.command == "aggregate-seeds":
        from s2r_det.evaluation.statistics import seed_summary

        return save_json(a.output, {k: seed_summary(v) for k, v in load_json(a.records).items()})
    if a.command == "bootstrap":
        from s2r_det.evaluation.statistics import (
            paired_sequence_bootstrap,
            paired_bootstrap,
            replicate_coco,
        )

        b = load_json(a.baseline)
        r = load_json(a.repaired)
        if a.unit == "sequence":
            return save_json(a.output, paired_sequence_bootstrap(b, r, a.replicates, a.seed))
        if a.annotations is None:
            raise ValueError("image AP bootstrap requires source COCO annotations")
        from s2r_det.evaluation.coco import standard_metrics

        ds = load_json(a.annotations)
        if isinstance(b, dict):
            b = b["coco_predictions"]
        if isinstance(r, dict):
            r = r["coco_predictions"]

        def metric(pred, ids):
            sample, predictions = replicate_coco(ds, pred, ids)
            return standard_metrics(sample, predictions)["APsmall"]

        return save_json(
            a.output,
            paired_bootstrap(
                [str(i["id"]) for i in ds["images"]],
                lambda ids: metric(b, ids),
                lambda ids: metric(r, ids),
                a.replicates,
                a.seed,
            ),
        )
    if a.command == "deployment":
        import importlib
        from s2r_det.deployment.validation import benchmark

        module, name = a.factory.split(":", 1)
        profile = getattr(importlib.import_module(module), name)()
        return save_json(
            a.output,
            benchmark(
                profile["predict"],
                profile["decoded_images"],
                a.samples,
                a.warmup,
                profile.get("synchronize", lambda: None),
            ),
        )
    raise ValueError("unhandled command")
