import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.structures.bbox import distance2bbox, get_box_tensor


EPS = 1e-12
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 0
STABLE_FREQ = 0.60


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json_yaml(path):
    # eligible_layers.yaml was intentionally emitted as YAML-1.2-compatible JSON.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def raw_valid_line_to_index(path):
    result = {}
    idx = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            p = line.strip().rstrip(",").split(",")
            if len(p) < 8:
                continue

            x, y, w, h = map(float, p[:4])
            score = int(float(p[4]))
            cat = int(float(p[5]))

            if score == 1 and 1 <= cat <= 10 and w > 0 and h > 0:
                result[line_no] = idx
                idx += 1

    return result, idx


def make_dataset(cfg, data_root, train_json):
    ds_cfg = copy.deepcopy(cfg.val_dataloader.dataset)

    ds_cfg.data_root = str(Path(data_root)) + "/"
    ds_cfg.ann_file = str(
        Path(train_json).relative_to(Path(data_root))
    )
    ds_cfg.data_prefix = dict(
        img="VisDrone2019-DET-train/images/"
    )
    ds_cfg.test_mode = True

    return DATASETS.build(ds_cfg)


def flatten_targets(head, cls_scores, bbox_preds, sample):
    num_imgs = 1
    metas = [dict(sample.metainfo)]

    featmap_sizes = [x.shape[-2:] for x in cls_scores]
    device = cls_scores[0].device

    anchor_list, valid_flag_list = head.get_anchors(
        featmap_sizes, metas, device=device
    )

    # Save pre-mutation flat anchor/valid map because get_targets mutates lists.
    flat_anchors = torch.cat(anchor_list[0], dim=0)
    flat_valid = torch.cat(valid_flag_list[0], dim=0)

    inside = anchor_inside_flags(
        flat_anchors,
        flat_valid,
        metas[0]["img_shape"][:2],
        head.train_cfg["allowed_border"],
    )

    flatten_cls = torch.cat([
        x.permute(0, 2, 3, 1).reshape(
            num_imgs, -1, head.cls_out_channels
        )
        for x in cls_scores
    ], dim=1)

    decoded = []

    for anchor, pred in zip(anchor_list[0], bbox_preds):
        a = anchor.reshape(-1, 4)
        p = pred.permute(0, 2, 3, 1).reshape(
            num_imgs, -1, 4
        )
        decoded.append(distance2bbox(a, p))

    flatten_bbox = torch.cat(decoded, dim=1)

    gt_instances = [sample.gt_instances]

    targets = head.get_targets(
        flatten_cls,
        flatten_bbox,
        anchor_list,
        valid_flag_list,
        gt_instances,
        metas,
        batch_gt_instances_ignore=None,
    )

    if targets is None:
        raise RuntimeError("head.get_targets returned None")

    (
        _anchors,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        assign_metrics_list,
        sampling_results,
    ) = targets

    labels = torch.cat([
        x.reshape(-1) for x in labels_list
    ], dim=0)

    label_weights = torch.cat([
        x.reshape(-1) for x in label_weights_list
    ], dim=0)

    bbox_targets = torch.cat([
        x.reshape(-1, 4) for x in bbox_targets_list
    ], dim=0)

    assign_metrics = torch.cat([
        x.reshape(-1) for x in assign_metrics_list
    ], dim=0)

    sampling = sampling_results[0]

    inside_global = torch.nonzero(
        inside, as_tuple=False
    ).flatten()

    full_pos_inds = inside_global[sampling.pos_inds]

    return {
        "flatten_cls": flatten_cls[0],
        "flatten_bbox": flatten_bbox[0],
        "labels": labels,
        "label_weights": label_weights,
        "bbox_targets": bbox_targets,
        "assign_metrics": assign_metrics,
        "sampling": sampling,
        "full_pos_inds": full_pos_inds,
    }


def instance_loss(head, packed, gt_index):
    sampling = packed["sampling"]

    assigned = sampling.pos_assigned_gt_inds
    mask = assigned == int(gt_index)

    local_pos = sampling.pos_inds[mask]
    full_pos = packed["full_pos_inds"][mask]

    if len(full_pos) == 0:
        return None, 0

    labels = packed["labels"][full_pos]
    metrics = packed["assign_metrics"][full_pos]

    cls_logits = packed["flatten_cls"][full_pos]
    pred_boxes = packed["flatten_bbox"][full_pos]
    target_boxes = packed["bbox_targets"][full_pos]

    # Same alignment metric weighting as RTMDet, but normalized per GT.
    norm = max(float(metrics.detach().sum().item()), 1e-12)

    cls_weight = torch.ones_like(metrics)

    loss_cls = head.loss_cls(
        cls_logits,
        (labels, metrics),
        cls_weight,
        avg_factor=norm,
    )

    loss_bbox = head.loss_bbox(
        pred_boxes,
        target_boxes,
        weight=metrics,
        avg_factor=norm,
    )

    return loss_cls + loss_bbox, int(len(full_pos))



def flatten_dense_predictions(head, cls_scores, bbox_preds, sample):
    """Raw dense RTMDet predictions before score filtering/NMS."""
    metas = [dict(sample.metainfo)]
    featmap_sizes = [x.shape[-2:] for x in cls_scores]
    device = cls_scores[0].device

    anchor_list, _ = head.get_anchors(
        featmap_sizes,
        metas,
        device=device,
    )

    flatten_cls = torch.cat([
        x.permute(0, 2, 3, 1).reshape(
            1, -1, head.cls_out_channels
        )
        for x in cls_scores
    ], dim=1)[0]

    decoded = []

    for anchor, pred in zip(anchor_list[0], bbox_preds):
        a = anchor.reshape(-1, 4)
        pp = pred.permute(0, 2, 3, 1).reshape(
            1, -1, 4
        )
        decoded.append(
            distance2bbox(a, pp)[0]
        )

    flatten_bbox = torch.cat(decoded, dim=0)

    return flatten_cls, flatten_bbox


def dense_best_iou_instance_loss(
    head,
    flatten_cls,
    flatten_bbox,
    gt_box,
    gt_label,
):
    """
    Deterministic per-GT matching independent of training assignment.

    Match the GT to the decoded dense prediction with maximum IoU.
    Matching index is treated as a fixed discrete correspondence;
    gradients flow through the selected classification and box outputs.
    """
    if len(flatten_bbox) == 0:
        return None, -1, float("nan")

    with torch.no_grad():
        lt = torch.maximum(
            flatten_bbox[:, :2],
            gt_box[None, :2],
        )
        rb = torch.minimum(
            flatten_bbox[:, 2:],
            gt_box[None, 2:],
        )

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        pa = (
            (flatten_bbox[:, 2] - flatten_bbox[:, 0]).clamp(min=0)
            * (flatten_bbox[:, 3] - flatten_bbox[:, 1]).clamp(min=0)
        )

        ga = (
            (gt_box[2] - gt_box[0]).clamp(min=0)
            * (gt_box[3] - gt_box[1]).clamp(min=0)
        )

        iou = inter / (
            pa + ga - inter
        ).clamp(min=1e-12)

        # torch.argmax returns the first index on ties -> deterministic.
        match_idx = int(torch.argmax(iou).item())
        match_iou = float(iou[match_idx].item())

    pred_logits = flatten_cls[
        match_idx:match_idx + 1
    ]

    pred_box = flatten_bbox[
        match_idx:match_idx + 1
    ]

    label = torch.tensor(
        [int(gt_label)],
        dtype=torch.long,
        device=pred_logits.device,
    )

    # Ideal GT-associated quality target.
    quality = torch.ones(
        1,
        dtype=pred_logits.dtype,
        device=pred_logits.device,
    )

    cls_weight = torch.ones(
        1,
        dtype=pred_logits.dtype,
        device=pred_logits.device,
    )

    loss_cls = head.loss_cls(
        pred_logits,
        (label, quality),
        cls_weight,
        avg_factor=1.0,
    )

    bbox_weight = torch.ones(
        1,
        dtype=pred_box.dtype,
        device=pred_box.device,
    )

    loss_bbox = head.loss_bbox(
        pred_box,
        gt_box[None],
        weight=bbox_weight,
        avg_factor=1.0,
    )

    return (
        loss_cls + loss_bbox,
        match_idx,
        match_iou,
    )


def sensitivity_for_loss(loss, activation_map, layer_names):
    tensors = []
    owner = []

    for li, name in enumerate(layer_names):
        for a in activation_map[name]:
            tensors.append(a)
            owner.append(li)

    grads = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    num = np.zeros(len(layer_names), dtype=np.float64)
    den2 = np.zeros(len(layer_names), dtype=np.float64)
    count = np.zeros(len(layer_names), dtype=np.int64)

    for a, g, li in zip(tensors, grads, owner):
        aa = a.float()

        n = aa.numel()
        den2[li] += float(
            aa.detach().pow(2).sum().item()
        )
        count[li] += n

        if g is not None:
            num[li] += float(
                (aa * g.float()).detach().abs().sum().item()
            )

    out = np.zeros(len(layer_names), dtype=np.float64)

    for i in range(len(layer_names)):
        if count[i] <= 0:
            out[i] = np.nan
            continue

        numerator = num[i] / float(count[i])
        denominator = math.sqrt(
            den2[i] / float(count[i])
        ) + EPS

        out[i] = numerator / denominator

    return out


def weighted_bootstrap(records, layer_names, analysis_meta):
    g = np.asarray(
        [r["g"] for r in records],
        dtype=np.float64
    )

    logg = np.log(g + EPS)

    failure_mask = np.asarray([
        r["cohort"] == "failure"
        for r in records
    ])
    success_mask = np.asarray([
        r["cohort"] == "retention"
        for r in records
    ])

    if failure_mask.sum() == 0 or success_mask.sum() == 0:
        raise RuntimeError("empty primary FCSL cohort")

    observed = (
        np.median(logg[failure_mask], axis=0)
        - np.median(logg[success_mask], axis=0)
    )

    rec_by_image = defaultdict(list)

    for idx, r in enumerate(records):
        rec_by_image[
            (r["cohort"], r["file_name"])
        ].append(idx)

    strata = defaultdict(list)

    for fn, meta in analysis_meta.items():
        strata[
            (meta["cohort"], meta["density"])
        ].append(fn)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    q_boot = np.zeros(
        (BOOTSTRAPS, len(layer_names)),
        dtype=np.float64
    )

    top1 = np.zeros(len(layer_names), dtype=np.int64)
    top2 = np.zeros(len(layer_names), dtype=np.int64)

    for b in range(BOOTSTRAPS):
        f_indices = []
        s_indices = []

        for (cohort, density), names in sorted(strata.items()):
            sampled = rng.choice(
                np.asarray(names, dtype=object),
                size=len(names),
                replace=True,
            )

            target = (
                f_indices
                if cohort == "failure"
                else s_indices
            )

            for fn in sampled:
                target.extend(
                    rec_by_image[(cohort, str(fn))]
                )

        if not f_indices or not s_indices:
            raise RuntimeError(
                "bootstrap produced empty cohort"
            )

        qb = (
            np.median(logg[f_indices], axis=0)
            - np.median(logg[s_indices], axis=0)
        )

        q_boot[b] = qb

        rank = np.argsort(-qb, kind="stable")
        top1[rank[0]] += 1
        top2[rank[:2]] += 1

    med = np.median(q_boot, axis=0)
    lo = np.quantile(q_boot, 0.025, axis=0)
    hi = np.quantile(q_boot, 0.975, axis=0)

    return {
        "observed": observed,
        "bootstrap_median": med,
        "ci_low": lo,
        "ci_high": hi,
        "top1_freq": top1 / BOOTSTRAPS,
        "top2_freq": top2 / BOOTSTRAPS,
        "q_boot": q_boot,
    }


def controls(
    eligible_layers,
    layer_names,
    bootstrap_result,
):
    by_name = {
        x["name"]: x for x in eligible_layers
    }

    q = bootstrap_result["bootstrap_median"]
    f1 = bootstrap_result["top1_freq"]
    f2 = bootstrap_result["top2_freq"]

    stable_top1 = [
        i for i in range(len(layer_names))
        if f1[i] >= STABLE_FREQ
    ]

    stable_top2 = [
        i for i in range(len(layer_names))
        if f2[i] >= STABLE_FREQ
    ]

    stable_top1.sort(
        key=lambda i: (-q[i], layer_names[i])
    )
    stable_top2.sort(
        key=lambda i: (-q[i], layer_names[i])
    )

    fcsl_top1 = (
        [layer_names[stable_top1[0]]]
        if stable_top1 else []
    )

    fcsl_top2 = (
        [layer_names[i] for i in stable_top2[:2]]
        if len(stable_top2) >= 2 else []
    )

    groups = defaultdict(list)

    for x in eligible_layers:
        groups[x["module_group"]].append(x["name"])

    # eligible_layers order follows model.named_modules(), so the final item
    # in a group is the deepest eligible layer in that structural group.
    manual_neck_p3 = [
        groups["neck_p3_high_resolution"][-1]
    ]

    manual_head = [
        groups["classification_tower"][-1],
        groups["regression_tower"][-1],
    ]

    manual_neck_head = list(dict.fromkeys(
        manual_neck_p3 + manual_head
    ))

    if fcsl_top2:
        reference_name = "FCSL_TOP2"
        reference = fcsl_top2
    elif fcsl_top1:
        reference_name = "FCSL_TOP1"
        reference = fcsl_top1
    else:
        reference_name = "MANUAL_NECK_P3"
        reference = manual_neck_p3

    def params(names):
        return int(sum(
            by_name[n]["flcr_rank8_parameters"]
            for n in names
        ))

    target_params = params(reference)
    k = len(reference)

    forbidden = {
        tuple(sorted(reference)),
        tuple(sorted(manual_neck_p3)),
        tuple(sorted(manual_head)),
        tuple(sorted(manual_neck_head)),
    }

    exact = []

    for comb in itertools.combinations(layer_names, k):
        key = tuple(sorted(comb))

        if key in forbidden:
            continue

        if params(comb) == target_params:
            exact.append(tuple(comb))

    # Deterministic three random equal-parameter controls.
    rng = random.Random(0)
    rng.shuffle(exact)

    random_sets = exact[:3]

    random_gate = len(random_sets) == 3

    return {
        "stable_any": bool(stable_top1 or stable_top2),
        "stable_top1_layers": [
            layer_names[i] for i in stable_top1
        ],
        "stable_top2_layers": [
            layer_names[i] for i in stable_top2
        ],
        "fcsl_top1": fcsl_top1,
        "fcsl_top2": fcsl_top2,
        "manual": {
            "Neck-P3": manual_neck_p3,
            "Head": manual_head,
            "Neck-P3+Head": manual_neck_head,
        },
        "random_reference": reference_name,
        "random_reference_layers": reference,
        "random_reference_params_rank8": target_params,
        "random_equal_parameter_sets": [
            {
                "id": f"Random-{i+1}",
                "layers": list(x),
                "rank8_parameters": params(x),
                "parameter_delta": params(x) - target_params,
            }
            for i, x in enumerate(random_sets)
        ],
        "random_equal_parameter_gate":
            "PASS" if random_gate else "HOLD",
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eligible", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--atlas", required=True)

    p.add_argument("--data-root", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--train-raw", required=True)

    p.add_argument("--output", required=True)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--skip-bootstrap", action="store_true")

    args = p.parse_args()

    t0 = time.time()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    eligible_doc = load_json_yaml(args.eligible)
    eligible_layers = eligible_doc["eligible_layers"]
    layer_names = [x["name"] for x in eligible_layers]

    cfg = Config.fromfile(args.config)

    model = init_detector(
        args.config,
        args.checkpoint,
        device="cuda:0"
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    dataset = make_dataset(
        cfg,
        args.data_root,
        args.train_json
    )

    name_to_idx = {}

    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        name_to_idx[Path(info["img_path"]).name] = i

    analysis_rows = []

    with open(args.analysis, newline="", encoding="utf-8") as f:
        analysis_rows = list(csv.DictReader(f))

    if len(analysis_rows) != 512:
        raise RuntimeError(
            f"analysis set expected 512 rows, got {len(analysis_rows)}"
        )

    if args.max_images > 0:
        analysis_rows = analysis_rows[:args.max_images]

    analysis_meta = {
        r["file_name"]: {
            "cohort": r["cohort"],
            "density": r["density"],
        }
        for r in analysis_rows
    }

    atlas = pq.read_table(args.atlas).to_pylist()

    target_by_image = defaultdict(list)

    selected_names = set(analysis_meta)

    for r in atlas:
        fn = r["file_name"]

        if fn not in selected_names:
            continue

        cohort = analysis_meta[fn]["cohort"]

        if (
            cohort == "failure"
            and r["failure_type"] in {"F1", "F2", "F3", "F4"}
        ):
            target_by_image[fn].append(r)

        elif (
            cohort == "retention"
            and r["failure_type"] == "F5"
        ):
            target_by_image[fn].append(r)

    for fn in selected_names:
        if not target_by_image[fn]:
            raise RuntimeError(
                f"selected image has no primary instance: {fn}"
            )

    modules = dict(model.named_modules())

    missing = [
        x for x in layer_names
        if x not in modules
    ]

    if missing:
        raise RuntimeError(
            "eligible modules missing: " + repr(missing)
        )

    activation_map = {
        x: [] for x in layer_names
    }

    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"non-tensor eligible output: {name}"
                )
            activation_map[name].append(output)
        return hook

    for name in layer_names:
        hooks.append(
            modules[name].register_forward_hook(
                make_hook(name)
            )
        )

    records = []

    mapping_mismatch = 0
    zero_assignment = 0
    nonfinite_loss = 0
    nonfinite_g = 0

    torch.cuda.reset_peak_memory_stats()

    for img_no, row in enumerate(analysis_rows, 1):
        fn = row["file_name"]
        cohort = row["cohort"]

        idx = name_to_idx.get(fn)

        if idx is None:
            raise RuntimeError(
                f"dataset index missing: {fn}"
            )

        item = dataset[idx]
        batch = pseudo_collate([item])

        processed = model.data_preprocessor(
            batch,
            training=False
        )

        inputs = processed["inputs"]
        sample = processed["data_samples"][0]

        inputs.requires_grad_(True)

        for k in activation_map:
            activation_map[k].clear()

        feats = model.extract_feat(inputs)
        cls_scores, bbox_preds = model.bbox_head(feats)

        dense_cls, dense_bbox = flatten_dense_predictions(
            model.bbox_head,
            cls_scores,
            bbox_preds,
            sample,
        )

        for name in layer_names:
            if not activation_map[name]:
                raise RuntimeError(
                    f"eligible activation missing: {name}"
                )

        # The val/test pipeline keeps gt_instances.bboxes in ORIGINAL
        # image coordinates even though model inputs are resized to 640.
        #
        # Keep an original-coordinate copy for E00 identity verification,
        # then rescale only the assignment copy to model-input coordinates.
        gt_boxes_original = get_box_tensor(
            sample.gt_instances.bboxes
        ).detach().clone()

        gt_labels_original = (
            sample.gt_instances.labels.detach().clone()
        )

        sf = sample.metainfo.get(
            "scale_factor", (1.0, 1.0)
        )
        sx = float(sf[0])
        sy = float(sf[1])

        # Per-instance FCSL uses singleton native RTMDet assignment.
        # GT identity remains checked in original coordinates below.
        # For each target GT j, a one-GT copy is transformed into model
        # input coordinates and passed to the unchanged RTMDet assigner.
        raw_path = (
            Path(args.train_raw)
            / "annotations"
            / (Path(fn).stem + ".txt")
        )

        line_to_gt, valid_raw_count = (
            raw_valid_line_to_index(raw_path)
        )

        gt_boxes = gt_boxes_original
        gt_labels = gt_labels_original

        if valid_raw_count != len(gt_labels):
            raise RuntimeError(
                f"raw/dataset GT count mismatch {fn}: "
                f"{valid_raw_count} vs {len(gt_labels)}"
            )

        for arow in target_by_image[fn]:
            line_no = int(arow["line_no"])

            if line_no not in line_to_gt:
                mapping_mismatch += 1
                continue

            gj = int(line_to_gt[line_no])

            expected_label = int(arow["category_id"]) - 1

            if int(gt_labels[gj].item()) != expected_label:
                mapping_mismatch += 1
                continue

            expected_box = torch.tensor(
                [
                    float(arow["gt_x1"]),
                    float(arow["gt_y1"]),
                    float(arow["gt_x2"]),
                    float(arow["gt_y2"]),
                ],
                device=gt_boxes.device,
                dtype=gt_boxes.dtype,
            )

            box_diff = float(
                (expected_box - gt_boxes[gj]).abs().max().item()
            )

            if box_diff > 1e-3:
                mapping_mismatch += 1
                continue

            input_gt_box = expected_box * expected_box.new_tensor(
                [sx, sy, sx, sy]
            )

            loss, dense_match_idx, dense_match_iou = (
                dense_best_iou_instance_loss(
                    model.bbox_head,
                    dense_cls,
                    dense_bbox,
                    input_gt_box,
                    expected_label,
                )
            )

            if loss is None or dense_match_idx < 0:
                zero_assignment += 1
                continue

            npos = 1

            loss_val = float(loss.detach().item())

            if not math.isfinite(loss_val):
                nonfinite_loss += 1
                continue

            g = sensitivity_for_loss(
                loss,
                activation_map,
                layer_names
            )

            if not np.all(np.isfinite(g)):
                nonfinite_g += 1
                continue

            rec = {
                "file_name": fn,
                "cohort": cohort,
                "density": row["density"],
                "failure_type": arow["failure_type"],
                "category_id": int(arow["category_id"]),
                "scale_group": arow["scale_group"],
                "line_no": line_no,
                "gt_index": gj,
                "positive_priors": npos,
                "dense_match_index": int(dense_match_idx),
                "dense_match_iou": float(dense_match_iou),
                "instance_loss": loss_val,
                "g": g,
            }

            records.append(rec)

        # Release graph image by image.
        del feats, cls_scores, bbox_preds, dense_cls, dense_bbox, inputs
        torch.cuda.empty_cache()

        if img_no % 10 == 0 or img_no == len(analysis_rows):
            print(
                f"PROGRESS={img_no}/{len(analysis_rows)} "
                f"instances={len(records)} "
                f"elapsed_min={(time.time()-t0)/60:.2f}",
                flush=True
            )

    for h in hooks:
        h.remove()

    expected_primary = sum(
        len(target_by_image[x["file_name"]])
        for x in analysis_rows
    )

    valid_engineering = (
        mapping_mismatch == 0
        and zero_assignment == 0
        and nonfinite_loss == 0
        and nonfinite_g == 0
        and len(records) == expected_primary
    )

    # --------------------------------------------------------
    # Save per-instance sensitivity as compact wide parquet.
    # --------------------------------------------------------
    parquet_rows = []

    for r in records:
        x = {
            k: v for k, v in r.items()
            if k != "g"
        }

        for i, value in enumerate(r["g"]):
            x[f"g_{i:03d}"] = float(value)

        parquet_rows.append(x)

    pq.write_table(
        pa.Table.from_pylist(parquet_rows),
        out / "instance_sensitivity.parquet",
        compression="zstd"
    )

    pq.read_table(out / "instance_sensitivity.parquet")

    Path(out / "layer_index.json").write_text(
        json.dumps(
            {
                f"g_{i:03d}": name
                for i, name in enumerate(layer_names)
            },
            indent=2
        ),
        encoding="utf-8"
    )

    result = {
        "processed_images": len(analysis_rows),
        "expected_primary_instances": expected_primary,
        "computed_instances": len(records),
        "failure_instances": sum(
            r["cohort"] == "failure"
            for r in records
        ),
        "retention_instances": sum(
            r["cohort"] == "retention"
            for r in records
        ),
        "mapping_mismatch": mapping_mismatch,
        "zero_positive_assignment": zero_assignment,
        "nonfinite_loss": nonfinite_loss,
        "nonfinite_sensitivity": nonfinite_g,
        "epsilon": EPS,
        "matching_method": "dense_best_iou_pre_threshold",
        "dense_match_iou_min": (
            float(min(r["dense_match_iou"] for r in records))
            if records else float("nan")
        ),
        "dense_match_iou_median": (
            float(np.median([
                r["dense_match_iou"] for r in records
            ]))
            if records else float("nan")
        ),
        "instance_loss_rule": (
            "Per-GT dense best-IoU matching independent of training "
            "assignment; selected raw decoded prediction uses native "
            "QualityFocalLoss with ideal GT quality target 1 plus "
            "native GIoULoss; no TaskAligned center/top-k constraint."
        ),
        "peak_vram_mb": int(
            torch.cuda.max_memory_allocated() / 1024 / 1024
        ),
        "wall_time_seconds": time.time() - t0,
        "engineering_gate":
            "PASS" if valid_engineering else "HOLD",
    }

    if args.skip_bootstrap:
        Path(out / "run_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8"
        )

        print(
            "SENSITIVITY_ENGINEERING_GATE="
            + result["engineering_gate"]
        )
        return

    if len(analysis_rows) != 512:
        raise RuntimeError(
            "bootstrap is only valid for frozen full 512-image set"
        )

    if not valid_engineering:
        raise RuntimeError(
            "engineering gate failed before bootstrap"
        )

    boot = weighted_bootstrap(
        records,
        layer_names,
        analysis_meta
    )

    ranks = np.argsort(
        -boot["bootstrap_median"],
        kind="stable"
    )

    layer_rows = []

    for rank, i in enumerate(ranks, 1):
        layer_rows.append({
            "rank": rank,
            "layer": layer_names[i],
            "q_observed": float(boot["observed"][i]),
            "q_bootstrap_median":
                float(boot["bootstrap_median"][i]),
            "q_ci_low": float(boot["ci_low"][i]),
            "q_ci_high": float(boot["ci_high"][i]),
            "top1_frequency": float(
                boot["top1_freq"][i]
            ),
            "top2_frequency": float(
                boot["top2_freq"][i]
            ),
            "stable_top1":
                bool(boot["top1_freq"][i] >= STABLE_FREQ),
            "stable_top2":
                bool(boot["top2_freq"][i] >= STABLE_FREQ),
        })

    with open(
        out / "layer_stability.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(layer_rows[0].keys())
        )
        w.writeheader()
        w.writerows(layer_rows)

    np.savez_compressed(
        out / "bootstrap_q.npz",
        q=boot["q_boot"],
        layers=np.asarray(layer_names, dtype=object),
    )

    ctrl = controls(
        eligible_layers,
        layer_names,
        boot
    )

    Path(out / "candidate_controls.json").write_text(
        json.dumps(ctrl, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    stage_gate = (
        valid_engineering
        and ctrl["random_equal_parameter_gate"] == "PASS"
    )

    sensitivity_status = (
        "STABLE"
        if ctrl["stable_any"]
        else "UNSTABLE_CONTINUE_MANUAL"
    )

    result.update({
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "strata": "cohort x density_group, image-level resampling",
            "stable_frequency_threshold": STABLE_FREQ,
        },
        "sensitivity_status": sensitivity_status,
        "fcsl_top1": ctrl["fcsl_top1"],
        "fcsl_top2": ctrl["fcsl_top2"],
        "stable_top1_layers": ctrl["stable_top1_layers"],
        "stable_top2_layers": ctrl["stable_top2_layers"],
        "random_equal_parameter_gate":
            ctrl["random_equal_parameter_gate"],
        "stage_gate":
            "PASS" if stage_gate else "HOLD",
    })

    Path(out / "run_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(
        "SENSITIVITY_ENGINEERING_GATE="
        + result["engineering_gate"]
    )
    print("S00_STAGE_GATE=" + result["stage_gate"])
    print("SENSITIVITY_STATUS=" + sensitivity_status)


if __name__ == "__main__":
    main()
