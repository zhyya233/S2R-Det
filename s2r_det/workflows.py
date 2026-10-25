"""Shared paper workflows. CLI --help does not import detector dependencies."""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from s2r_det.utils.config import load_recipe
from s2r_det.utils.io import load_json, save_json, file_sha256, fingerprint


def adapter_and_data(args):
    from s2r_det.data.coco import CocoImages

    ds = CocoImages(args.annotations, args.image_root)
    if getattr(args, "adapter_factory", None):
        from s2r_det.integrations.yolov8 import load_external_adapter

        adapter = load_external_adapter(
            args.adapter_factory,
            config=args.detector_config,
            checkpoint=args.checkpoint,
            device=args.device,
            category_ids=ds.categories,
        )
    else:
        from s2r_det.integrations.mmdetection import RTMDetAdapter

        adapter = RTMDetAdapter(args.detector_config, args.checkpoint, args.device, ds.categories)
    return adapter, ds


def identity(args, adapter):
    return {
        "baseline": file_sha256(args.checkpoint),
        "annotations": file_sha256(args.annotations),
        "preprocessing": fingerprint(
            {
                "geometry": "half-up aspect resize, bilinear PIL, upper-left pad114,640square",
                "detector_config": adapter.resolved_config_sha256,
            }
        ),
        "dense_convention": fingerprint(adapter.dense_convention),
        "atlas": file_sha256(args.atlas),
    }


def infer(args):
    import torch
    from s2r_det.data.coco import resize_geometry

    adapter, ds = adapter_and_data(args)
    adapter.model.eval()
    images = []
    predictions = []
    for key in sorted(ds.images):
        with torch.no_grad():
            pre, trace = adapter.candidates(adapter.prepare(ds, [key]))
        im = ds.images[key]
        _, _, sx, sy = resize_geometry(im["width"], im["height"])
        for p in pre:
            p["box"] = [v / (sx if i % 2 == 0 else sy) for i, v in enumerate(p["box"])]
        images.append({"image_id": key, "gt": ds.gt(key), "pre": pre, "trace": trace})
        for i in trace["post_indices"]:
            p = pre[i]
            x, y, x2, y2 = p["box"]
            predictions.append(
                {
                    "image_id": im["id"],
                    "category_id": p["category_id"],
                    "bbox": [x, y, x2 - x, y2 - y],
                    "score": p["score"],
                }
            )
    save_json(
        args.output,
        {
            "images": images,
            "coco_predictions": predictions,
            "annotations_sha256": file_sha256(args.annotations),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": adapter.resolved_config_sha256,
        },
    )


def atlas(args):
    from s2r_det.failure_atlas.atlas import build_atlas

    obj = load_json(args.predictions)
    if args.fit_training_quartiles and args.split != "train":
        raise ValueError("quartiles may only be fitted to labeled training images")
    thresholds = load_json(args.quartiles) if args.quartiles else None
    if thresholds is not None:
        thresholds = thresholds.get("quartiles", thresholds)
    out = build_atlas(obj["images"], thresholds, args.fit_training_quartiles)
    out["split"] = args.split
    out["prediction_artifact_sha256"] = file_sha256(args.predictions)
    for name in ["annotations_sha256", "checkpoint_sha256", "config_sha256"]:
        out[name] = obj[name]
    save_json(args.output, out)


def manifest(args):
    from s2r_det.sensitivity.manifest import build_manifest

    adapter, ds = adapter_and_data(args)
    adapter.model.eval()
    if args.adapter_factory:
        raise ValueError(
            "external architectures must supply their own explicit canonical semantic manifest"
        )
    inputs = adapter.prepare(ds, [sorted(ds.images)[0]])["inputs"]
    rows = build_manifest(
        adapter.model,
        inputs,
        load_json(args.insertion_sets),
        load_json(args.group_overrides) if args.group_overrides else None,
    )
    save_json(
        args.output,
        {
            "modules": rows,
            "baseline_sha256": file_sha256(args.checkpoint),
            "config_sha256": adapter.resolved_config_sha256,
        },
    )


def validate_inputs(args, adapter, ds):
    a = load_json(args.atlas)
    if a.get("split") != "train":
        raise ValueError("repair analysis/cache/sampling require a training Atlas")
    for key, expected in [
        ("annotations_sha256", file_sha256(args.annotations)),
        ("checkpoint_sha256", file_sha256(args.checkpoint)),
        ("config_sha256", adapter.resolved_config_sha256),
    ]:
        if a.get(key) != expected:
            raise ValueError(f"training Atlas provenance mismatch: {key}")
    return a


def sensitivity(args):
    from s2r_det.sensitivity.cohorts import build_cohorts
    from s2r_det.sensitivity.scoring import group_logs, bootstrap_stability, select_placement
    from s2r_det.sensitivity.autograd import ActivationRecorder, diagnostic_loss

    adapter, ds = adapter_and_data(args)
    a = validate_inputs(args, adapter, ds)
    m = load_json(args.manifest)
    if (
        m["baseline_sha256"] != file_sha256(args.checkpoint)
        or m["config_sha256"] != adapter.resolved_config_sha256
    ):
        raise ValueError("manifest baseline/config mismatch")
    rows = m["modules"]
    cohorts = build_cohorts(a["records"], ds.counts())
    by_image = defaultdict(list)
    for r in a["records"]:
        if r["category"] != "ambiguous":
            by_image[str(r["image_id"])].append(r)
    adapter.model.requires_grad_(False)
    adapter.model.eval()
    records = []
    with ActivationRecorder(
        adapter.model, [m["path"] for m in rows if m["analysis_eligible"]]
    ) as recorder:
        for image_id in sorted(cohorts["images"]):
            recorder.clear()
            prepared = adapter.prepare(ds, [image_id])
            prepared["inputs"].requires_grad_(True)
            logits, boxes, _ = adapter.dense(prepared)
            meta = prepared["data_samples"][0].metainfo
            sx, sy = meta["scale_factor"]
            for r in by_image[image_id]:
                adapter.model.zero_grad(set_to_none=True)
                gt = boxes.new_tensor(r["box"]) * boxes.new_tensor([sx, sy, sx, sy])
                objective = getattr(adapter, "diagnostic_loss", None)
                loss, _ = (
                    objective(logits[0], boxes[0], gt, ds.labels[r["category_id"]])
                    if objective
                    else diagnostic_loss(
                        adapter.head, logits[0], boxes[0], gt, ds.labels[r["category_id"]]
                    )
                )
                records.append(
                    {
                        "image_id": image_id,
                        "annotation_id": r["annotation_id"],
                        "category": r["category"],
                        "h": group_logs(recorder.sensitivities(loss), rows),
                    }
                )
    result = bootstrap_stability(records, cohorts["images"])
    selection = select_placement(result, rows, args.budget, args.rank, args.groups)
    save_json(
        args.output,
        {
            "method": "FCSL",
            "statistics": result,
            "selection": selection,
            "cohorts": cohorts,
            "manifest_sha256": file_sha256(args.manifest),
            "baseline_sha256": file_sha256(args.checkpoint),
            "atlas_sha256": file_sha256(args.atlas),
        },
    )


def cache(args):
    import torch
    from s2r_det.retention.cache import build_cache_entry

    adapter, ds = adapter_and_data(args)
    a = validate_inputs(args, adapter, ds)
    adapter.model.eval()
    entries = {}
    for image_id in a["retention_pool"]:
        prepared = adapter.prepare(ds, [image_id])
        sx, sy = prepared["data_samples"][0].metainfo["scale_factor"]
        gt = []
        for r in a["records"]:
            if r["image_id"] == image_id and r["category"] == "F5":
                gt.append(
                    {
                        **r,
                        "category_id": ds.labels[r["category_id"]],
                        "box": [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(r["box"])],
                    }
                )
        with torch.no_grad():
            logits, boxes, _ = adapter.dense(prepared)
        entries[image_id] = build_cache_entry(image_id, logits[0], boxes[0], gt)
    save_json(
        args.output,
        {
            "identity": identity(args, adapter),
            "eligible_image_ids": a["retention_pool"],
            "entries": entries,
        },
    )


def train(args):
    import copy
    import torch
    from s2r_det.training.protocol import RepairProtocol, seed_everything
    from s2r_det.training.variants import configure_variant
    from s2r_det.training.engine import repair
    from s2r_det.sampling.streams import (
        FailureStratifiedBatchSource,
        bounded_failure_quotas,
        uniform_batches,
    )
    from s2r_det.adapters.flcr import fuse_flcr_inplace, adapter_parameter_count

    recipe = load_recipe(args.recipe)
    if recipe["method"] == "frozen":
        raise ValueError("Frozen baseline is evaluation-only, not a repair-training recipe")
    if recipe.get("requires_external_control_factory"):
        raise ValueError(recipe["requires_external_control_factory"])
    if recipe["method"] == "s2r_det" and any(
        not getattr(args, k, None) for k in ["atlas", "manifest", "selection", "retention_cache"]
    ):
        raise ValueError(
            "complete method requires a training Atlas, semantic manifest, actual FCSL selection and teacher cache"
        )
    if args.selection and (not args.atlas or not args.manifest):
        raise ValueError("FCSL selection must bind the supplied Atlas and manifest")
    if args.seed not in recipe.get("allowed_seeds", [0, 1, 2]):
        raise ValueError("seed not permitted by this named experiment protocol")
    protocol_values = dict(recipe["protocol"])
    if args.learning_rate is not None:
        protocol_values["learning_rate"] = args.learning_rate
    if protocol_values.get("learning_rate") is None:
        raise ValueError(
            "learning-rate family unspecified for this control; supply --learning-rate explicitly"
        )
    protocol = RepairProtocol(**protocol_values, seed=args.seed)
    seed_everything(args.seed)
    adapter, ds = adapter_and_data(args)
    a = validate_inputs(args, adapter, ds) if args.atlas else None
    paths = None
    if recipe.get("uses_flcr"):
        if args.selection:
            selection = load_json(args.selection)
            if (
                selection.get("method") != "FCSL"
                or selection["baseline_sha256"] != file_sha256(args.checkpoint)
                or selection["atlas_sha256"] != file_sha256(args.atlas)
                or selection["manifest_sha256"] != file_sha256(args.manifest)
            ):
                raise ValueError("FCSL selection identity mismatch")
            if selection["selection"]["rank"] != recipe["rank"]:
                raise ValueError("selection/repair rank mismatch")
            paths = selection["selection"]["paths"]
        elif recipe["method"] == "s2r_det":
            raise ValueError(
                "complete S2R-Det requires actual FCSL selection, not a manual Head default"
            )
        elif args.placement:
            paths = load_json(args.placement)["paths"]
        else:
            raise ValueError("manual/control FLCR requires an externally specified placement")
    names = configure_variant(
        adapter.model, recipe["method"], paths, recipe.get("rank", 8), recipe.get("norm_paths")
    )
    if (
        args.selection
        and adapter_parameter_count(adapter.model) != selection["selection"]["trainable_parameters"]
    ):
        raise ValueError("actual repair capacity does not match computed placement manifest")
    if recipe.get("expected_primary_trainable_parameters") is not None:
        actual = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
        if actual != recipe["expected_primary_trainable_parameters"]:
            raise ValueError(
                "documented primary capacity not satisfied by supplied architecture/sites"
            )
    sampling = recipe["sampling"]
    ids = sorted(ds.images)
    if sampling["kind"] == "failure_stratified":
        if a is None:
            raise ValueError("failure sampling requires training Atlas")
        batches = FailureStratifiedBatchSource(
            a["failure_pools"],
            a["retention_pool"],
            bounded_failure_quotas(a["failure_type_counts"]),
            protocol.batch_size,
            args.seed,
            sampling["failure_fraction"],
        ).iter_batches(protocol.steps)
    else:
        if sampling["kind"] in {"random_subset", "top_loss"}:
            support = load_json(args.support) if args.support else None
            if support is None:
                raise ValueError(
                    "subset/Top-loss control requires an explicit support record; historical selection is not invented"
                )
            if support.get("sampling_kind") != sampling["kind"]:
                raise ValueError("support manifest method does not match the sampling control")
            if sampling["kind"] == "top_loss" and not support.get("score_source_identity"):
                raise ValueError("Top-loss support requires a recorded score-source identity")
            ids = support["image_ids"]
            if not set(ids) <= set(ds.images):
                raise ValueError("subset contains unknown image IDs")
        elif sampling["kind"] != "uniform":
            raise ValueError("unknown image-sampling policy")
        batches = uniform_batches(ids, protocol.steps, protocol.batch_size, args.seed)
    teacher = load_json(args.retention_cache) if args.retention_cache else None
    receipt = repair(
        adapter, ds, batches, protocol, teacher, identity(args, adapter) if teacher else None
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        **receipt,
        "recipe": recipe,
        "trainable_names": names,
        "placement": paths,
        "baseline_sha256": file_sha256(args.checkpoint),
        "historical_equivalence": "UNVERIFIED",
    }
    from dataclasses import asdict

    metadata["effective_protocol"] = asdict(protocol)
    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "metadata": metadata,
            "format_version": 1,
            "fused": False,
        },
        output / "repair-final.pt",
    )
    fused_adapter = copy.deepcopy(adapter)
    fused_adapter.model.eval()
    adapter.model.eval()
    replaced = fuse_flcr_inplace(fused_adapter.model)
    if replaced:
        from s2r_det.deployment.validation import compare_outputs

        prepared = adapter.prepare(ds, [sorted(ds.images)[0]])
        with torch.no_grad():
            reference = adapter.dense(prepared)
            candidate = fused_adapter.dense(prepared)
        agreement = compare_outputs(reference[:2], candidate[:2])
        metadata["fusion_check"] = {
            **agreement,
            "checked_images": 1,
            "scope": "one supplied image pre-NMS tensors; not exhaustive deployment validation",
        }
        if not agreement["within_tolerance"]:
            raise RuntimeError(
                "fused pre-NMS output check failed; deployment checkpoint not written"
            )
    torch.save(
        {
            "state_dict": fused_adapter.model.state_dict(),
            "metadata": metadata,
            "format_version": 1,
            "fused": bool(replaced),
        },
        output / "deployment-final.pt",
    )
    save_json(output / "repair-metadata.json", metadata)


def evaluate(args):
    from s2r_det.evaluation.coco import standard_metrics, uavdt_metrics

    ds = load_json(args.annotations)
    obj = load_json(args.predictions)
    pred = obj.get("coco_predictions", obj) if isinstance(obj, dict) else obj
    if isinstance(obj, dict) and obj.get("annotations_sha256") != file_sha256(args.annotations):
        raise ValueError("prediction/annotation identity mismatch")
    out = (
        uavdt_metrics(ds, pred, args.max_detections)
        if args.dataset == "uavdt"
        else standard_metrics(ds, pred)
    )
    save_json(args.output, out)


def diagnostics(args):
    from s2r_det.evaluation.matching import transitions

    b = load_json(args.baseline)
    r = load_json(args.repaired)
    if (
        b["annotations_sha256"] != r["annotations_sha256"]
        or b["config_sha256"] != r["config_sha256"]
    ):
        raise ValueError("diagnostic populations or fixed inference configurations differ")
    bm = {i["image_id"]: i for i in b["images"]}
    rm = {i["image_id"]: i for i in r["images"]}
    if bm.keys() != rm.keys():
        raise ValueError("diagnostic image populations differ")
    all_gt = []
    for name in sorted(bm):
        x, y = bm[name], rm[name]
        if x["gt"] != y["gt"]:
            raise ValueError("GT changed between evaluations")
        # Cross-image overlap is forbidden: match each image independently.
        result = transitions(
            x["gt"],
            [x["pre"][i] for i in x["trace"]["post_indices"]],
            [y["pre"][i] for i in y["trace"]["post_indices"]],
        )
        all_gt.append({"image_id": name, **result})
    n = sum(x["GT"] for x in all_gt)
    recovered = sum(x["recovered"] for x in all_gt)
    damaged = sum(x["damaged"] for x in all_gt)
    success = sum(x["baseline_success"] for x in all_gt)
    save_json(
        args.output,
        {
            "GT": n,
            "recovered": recovered,
            "damaged": damaged,
            "net": recovered - damaged,
            "net_percent": 100 * (recovered - damaged) / n if n else None,
            "success_retention_percent": 100 * (success - damaged) / success if success else None,
            "unit": "all supplied retained GT; rates computed within one repair seed",
            "per_image": all_gt,
        },
    )
