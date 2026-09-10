import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mmcv.ops import batched_nms
from mmdet.apis import init_detector, inference_detector


CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(
            (len(a), len(b)), device=a.device, dtype=torch.float32
        )

    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    aa = (
        (a[:, 2] - a[:, 0]).clamp(min=0)
        * (a[:, 3] - a[:, 1]).clamp(min=0)
    )
    bb = (
        (b[:, 2] - b[:, 0]).clamp(min=0)
        * (b[:, 3] - b[:, 1]).clamp(min=0)
    )

    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-12)


def parse_gt(path, img_w, img_h):
    rows = []
    scale = min(640.0 / float(img_w), 640.0 / float(img_h))

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().rstrip(",").split(",")
            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            cat = int(float(parts[5]))
            trunc = int(float(parts[6]))
            occ = int(float(parts[7]))

            if not (
                score == 1
                and 1 <= cat <= 10
                and w > 0
                and h > 0
            ):
                continue

            area = w * h
            input_area = area * scale * scale
            is_small = area < 32.0 * 32.0
            is_vt16 = input_area < 16.0 * 16.0
            target = is_small or is_vt16

            rows.append({
                "line_no": line_no,
                "category_id": cat,
                "category_name": CATEGORIES[cat],
                "truncation": trunc,
                "occlusion": occ,
                "gt_x1": x,
                "gt_y1": y,
                "gt_x2": x + w,
                "gt_y2": y + h,
                "gt_w": w,
                "gt_h": h,
                "original_area": area,
                "input640_area": input_area,
                "is_coco_small": bool(is_small),
                "is_vt16": bool(is_vt16),
                "is_target": bool(target),
                "scale_group":
                    "VT16" if is_vt16 else "COCO_small_nonVT16",
            })

    return rows


def greedy_match_from_matrix(
    mat, gt_global, pred_global, min_iou=0.1
):
    match_idx = {}
    match_iou = {}

    if mat.numel() == 0:
        return match_idx, match_iou

    cand = torch.nonzero(mat >= min_iou, as_tuple=False)
    if cand.numel() == 0:
        return match_idx, match_iou

    vals = mat[cand[:, 0], cand[:, 1]].detach().cpu().numpy()
    cg = gt_global[cand[:, 0].detach().cpu().numpy()]
    cp = pred_global[cand[:, 1].detach().cpu().numpy()]

    order = np.lexsort((cp, cg, -vals))

    used_g = set()
    used_p = set()

    for z in order:
        g = int(cg[z])
        p = int(cp[z])
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        match_idx[g] = p
        match_iou[g] = float(vals[z])

    return match_idx, match_iou


def box_values(boxes_np, idx):
    if idx is None or idx < 0 or idx >= len(boxes_np):
        return [float("nan")] * 4
    return [float(v) for v in boxes_np[idx]]


def score_value(scores_np, idx):
    if idx is None or idx < 0 or idx >= len(scores_np):
        return float("nan")
    return float(scores_np[idx])


def density_group(n):
    if n <= 20:
        return "sparse"
    if n <= 50:
        return "medium"
    return "dense"


def write_atlas_csv(records, path):
    dims = [
        ("overall", lambda r: "all"),
        ("category", lambda r: r["category_name"]),
        ("scale_group", lambda r: r["scale_group"]),
        ("density_group", lambda r: r["density_group"]),
        ("occlusion", lambda r: str(r["occlusion"])),
        ("truncation", lambda r: str(r["truncation"])),
    ]

    labels = ["F1", "F2", "F3", "F4", "F5", "F-ambiguous"]
    out = []

    for dim_name, fn in dims:
        groups = defaultdict(list)
        for r in records:
            groups[fn(r)].append(r)

        for group, rs in sorted(groups.items(), key=lambda x: str(x[0])):
            total = len(rs)
            c = Counter(r["failure_type"] for r in rs)

            for label in labels:
                out.append({
                    "dimension": dim_name,
                    "group": group,
                    "failure_type": label,
                    "count": int(c[label]),
                    "group_total": int(total),
                    "rate": float(c[label] / total) if total else 0.0,
                })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dimension", "group", "failure_type",
                "count", "group_total", "rate"
            ]
        )
        w.writeheader()
        w.writerows(out)


def draw_cases(records, image_root, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    by_type = defaultdict(list)
    for r in records:
        by_type[r["failure_type"]].append(r)

    for label in ["F1", "F2", "F3", "F4", "F5", "F-ambiguous"]:
        pool = by_type.get(label, [])
        if not pool:
            continue

        picks = pool if len(pool) <= 2 else rng.sample(pool, 2)

        for j, r in enumerate(picks):
            src = Path(image_root) / r["file_name"]
            im = cv2.imread(str(src))
            if im is None:
                continue

            gx1, gy1, gx2, gy2 = [
                int(round(r[k]))
                for k in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]
            ]
            cv2.rectangle(im, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

            if label == "F4":
                keys = [
                    "suppressed_x1", "suppressed_y1",
                    "suppressed_x2", "suppressed_y2"
                ]
            elif label in {"F3", "F5"}:
                keys = [
                    "post_match_x1", "post_match_y1",
                    "post_match_x2", "post_match_y2"
                ]
            else:
                keys = [
                    "pre_best_x1", "pre_best_y1",
                    "pre_best_x2", "pre_best_y2"
                ]

            vals = [r[k] for k in keys]
            if all(math.isfinite(float(v)) for v in vals):
                x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

            txt = (
                f"{label} {r['category_name']} "
                f"pre={r['pre_best_iou']:.2f} "
                f"post={r['post_best_iou']:.2f}"
            )
            cv2.putText(
                im, txt, (max(0, gx1), max(20, gy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            dst = out_dir / (
                f"{label.replace('-', '')}_{j}_"
                f"{Path(r['file_name']).stem}.jpg"
            )
            cv2.imwrite(str(dst), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--dataset-sha", required=True)
    ap.add_argument("--checkpoint-sha", required=True)
    ap.add_argument("--config-sha", required=True)
    args = ap.parse_args()

    start = time.time()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.train_json, encoding="utf-8") as f:
        coco = json.load(f)

    images = sorted(coco["images"], key=lambda x: int(x["id"]))

    if len(images) != 6471:
        raise RuntimeError(
            f"Expected 6471 train images, got {len(images)}"
        )

    if args.max_images > 0:
        images = images[:args.max_images]

    model = init_detector(
        args.config, args.checkpoint, device="cuda:0"
    )
    model.eval()

    test_cfg = model.test_cfg
    nms_cfg = dict(test_cfg.nms)
    max_per_img = int(test_cfg.max_per_img)

    captures = {}

    def hook(module, inputs, output):
        captures["head"] = output

    handle = model.bbox_head.register_forward_hook(hook)

    records = []

    total_valid_gt = 0
    total_pre = 0
    total_post = 0

    reconstruction_mismatches = 0
    max_box_diff = 0.0
    max_score_diff = 0.0
    label_mismatch_total = 0

    # inference_detector(list) performs per-image forward in this MMDetection version; keep hook/output alignment explicit.
    batch_size = 1

    for st in range(0, len(images), batch_size):
        batch = images[st:st + batch_size]
        paths = [
            str(Path(args.train_root) / "images" / im["file_name"])
            for im in batch
        ]

        captures.clear()

        with torch.no_grad():
            results = inference_detector(model, paths)

        if not isinstance(results, list):
            results = [results]

        if "head" not in captures:
            raise RuntimeError("bbox_head output capture failed")

        metas = [dict(r.metainfo) for r in results]

        with torch.no_grad():
            pre_list = model.bbox_head.predict_by_feat(
                *captures["head"],
                batch_img_metas=metas,
                cfg=model.test_cfg,
                rescale=True,
                with_nms=False,
            )

        for im_meta, result, pre in zip(batch, results, pre_list):
            file_name = im_meta["file_name"]
            image_id = int(im_meta["id"])
            iw = int(im_meta["width"])
            ih = int(im_meta["height"])

            raw_path = (
                Path(args.train_root)
                / "annotations"
                / (Path(file_name).stem + ".txt")
            )
            gts = parse_gt(raw_path, iw, ih)

            total_valid_gt += len(gts)
            n_gt = len(gts)

            post = result.pred_instances
            total_pre += len(pre)
            total_post += len(post)

            dets, keep_all = batched_nms(
                pre.bboxes,
                pre.scores,
                pre.labels,
                nms_cfg
            )
            keep_post = keep_all[:max_per_img]

            rebuilt_boxes = pre.bboxes[keep_post]
            rebuilt_scores = pre.scores[keep_post]
            rebuilt_labels = pre.labels[keep_post]

            if (
                len(rebuilt_boxes) != len(post.bboxes)
                or len(rebuilt_labels) != len(post.labels)
            ):
                reconstruction_mismatches += 1
            else:
                if len(post.bboxes):
                    bd = float(
                        (rebuilt_boxes - post.bboxes).abs().max().item()
                    )
                    sd = float(
                        (rebuilt_scores - post.scores).abs().max().item()
                    )
                    lm = int(
                        (rebuilt_labels != post.labels).sum().item()
                    )

                    max_box_diff = max(max_box_diff, bd)
                    max_score_diff = max(max_score_diff, sd)
                    label_mismatch_total += lm

                    if bd > 1e-5 or sd > 1e-6 or lm != 0:
                        reconstruction_mismatches += 1

            if not gts:
                continue

            device = pre.bboxes.device

            gt_boxes = torch.tensor(
                [
                    [g["gt_x1"], g["gt_y1"], g["gt_x2"], g["gt_y2"]]
                    for g in gts
                ],
                dtype=torch.float32,
                device=device
            )
            gt_labels = torch.tensor(
                [g["category_id"] - 1 for g in gts],
                dtype=torch.long,
                device=device
            )

            npre = len(pre)

            suppressed_mask = torch.ones(
                npre, dtype=torch.bool, device=device
            )
            if len(keep_all):
                suppressed_mask[keep_all] = False

            truncated_mask = torch.zeros(
                npre, dtype=torch.bool, device=device
            )
            if len(keep_all) > max_per_img:
                truncated_mask[keep_all[max_per_img:]] = True

            post_source_pre = (
                keep_post.detach().cpu().numpy().astype(np.int64)
            )

            # Arrays indexed by all valid GT in this image.
            pre_best_iou = np.zeros(n_gt, dtype=np.float64)
            pre_best_idx = np.full(n_gt, -1, dtype=np.int64)
            post_best_iou = np.zeros(n_gt, dtype=np.float64)
            post_best_idx = np.full(n_gt, -1, dtype=np.int64)

            supp_best_iou = np.zeros(n_gt, dtype=np.float64)
            supp_best_idx = np.full(n_gt, -1, dtype=np.int64)

            trunc_best_iou = np.zeros(n_gt, dtype=np.float64)
            trunc_best_idx = np.full(n_gt, -1, dtype=np.int64)

            pre_match_iou = np.zeros(n_gt, dtype=np.float64)
            pre_match_idx = np.full(n_gt, -1, dtype=np.int64)
            post_match_iou = np.zeros(n_gt, dtype=np.float64)
            post_match_idx = np.full(n_gt, -1, dtype=np.int64)

            for cls in range(10):
                gi_t = torch.nonzero(
                    gt_labels == cls, as_tuple=False
                ).flatten()
                if gi_t.numel() == 0:
                    continue

                gi = gi_t.detach().cpu().numpy().astype(np.int64)

                # ---------------- Ppre ----------------
                pi_t = torch.nonzero(
                    pre.labels == cls, as_tuple=False
                ).flatten()

                if pi_t.numel():
                    pi = pi_t.detach().cpu().numpy().astype(np.int64)
                    mat = box_iou(gt_boxes[gi_t], pre.bboxes[pi_t])

                    vals, loc = mat.max(dim=1)
                    vals_np = vals.detach().cpu().numpy()
                    loc_np = loc.detach().cpu().numpy()

                    pre_best_iou[gi] = vals_np
                    pre_best_idx[gi] = pi[loc_np]

                    m_idx, m_iou = greedy_match_from_matrix(
                        mat, gi, pi, min_iou=0.1
                    )
                    for g, p in m_idx.items():
                        pre_match_idx[g] = p
                        pre_match_iou[g] = m_iou[g]

                    local_supp = suppressed_mask[pi_t]
                    if bool(local_supp.any()):
                        cols = torch.nonzero(
                            local_supp, as_tuple=False
                        ).flatten()
                        sm = mat[:, cols]
                        sv, sl = sm.max(dim=1)
                        sv_np = sv.detach().cpu().numpy()
                        sl_np = sl.detach().cpu().numpy()
                        global_cols = pi_t[cols].detach().cpu().numpy()
                        supp_best_iou[gi] = sv_np
                        supp_best_idx[gi] = global_cols[sl_np]

                    local_trunc = truncated_mask[pi_t]
                    if bool(local_trunc.any()):
                        cols = torch.nonzero(
                            local_trunc, as_tuple=False
                        ).flatten()
                        tm = mat[:, cols]
                        tv, tl = tm.max(dim=1)
                        tv_np = tv.detach().cpu().numpy()
                        tl_np = tl.detach().cpu().numpy()
                        global_cols = pi_t[cols].detach().cpu().numpy()
                        trunc_best_iou[gi] = tv_np
                        trunc_best_idx[gi] = global_cols[tl_np]

                # ---------------- Ppost ----------------
                pj_t = torch.nonzero(
                    post.labels == cls, as_tuple=False
                ).flatten()

                if pj_t.numel():
                    pj = pj_t.detach().cpu().numpy().astype(np.int64)
                    matp = box_iou(gt_boxes[gi_t], post.bboxes[pj_t])

                    vals, loc = matp.max(dim=1)
                    vals_np = vals.detach().cpu().numpy()
                    loc_np = loc.detach().cpu().numpy()

                    post_best_iou[gi] = vals_np
                    post_best_idx[gi] = pj[loc_np]

                    m_idx, m_iou = greedy_match_from_matrix(
                        matp, gi, pj, min_iou=0.1
                    )
                    for g, p in m_idx.items():
                        post_match_idx[g] = p
                        post_match_iou[g] = m_iou[g]

            pre_boxes_np = pre.bboxes.detach().cpu().numpy()
            pre_scores_np = pre.scores.detach().cpu().numpy()
            post_boxes_np = post.bboxes.detach().cpu().numpy()
            post_scores_np = post.scores.detach().cpu().numpy()

            keep_all_set = set(
                int(x) for x in keep_all.detach().cpu().tolist()
            )
            keep_post_set = set(int(x) for x in post_source_pre.tolist())

            dens = density_group(n_gt)

            for gi, g in enumerate(gts):
                if not g["is_target"]:
                    continue

                pb = int(pre_best_idx[gi])
                pmb = int(pre_match_idx[gi])
                qb = int(post_best_idx[gi])
                qmb = int(post_match_idx[gi])
                sb = int(supp_best_idx[gi])
                tb = int(trunc_best_idx[gi])

                if pb >= 0:
                    if pb not in keep_all_set:
                        survival = "nms_suppressed"
                    elif pb in keep_post_set:
                        survival = "post"
                    else:
                        survival = "maxper_truncated"
                else:
                    survival = "none"

                if qmb >= 0 and qmb < len(post_source_pre):
                    post_match_source_pre = int(post_source_pre[qmb])
                else:
                    post_match_source_pre = -1

                row = dict(g)
                row.update({
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_width": iw,
                    "image_height": ih,
                    "image_valid_gt_count": n_gt,
                    "density_group": dens,
                    "pre_count": int(len(pre)),
                    "post_count": int(len(post)),

                    "pre_best_idx": pb,
                    "pre_best_iou": float(pre_best_iou[gi]),
                    "pre_best_score": score_value(pre_scores_np, pb),
                    "pre_best_survival": survival,

                    "pre_match_idx": pmb,
                    "pre_match_iou": float(pre_match_iou[gi]),
                    "pre_match_score": score_value(pre_scores_np, pmb),

                    "post_best_idx": qb,
                    "post_best_iou": float(post_best_iou[gi]),
                    "post_best_score": score_value(post_scores_np, qb),

                    "post_match_idx": qmb,
                    "post_match_iou": float(post_match_iou[gi]),
                    "post_match_score": score_value(post_scores_np, qmb),
                    "post_match_source_pre_idx": post_match_source_pre,

                    "suppressed_best_pre_idx": sb,
                    "suppressed_best_iou": float(supp_best_iou[gi]),
                    "suppressed_best_score":
                        score_value(pre_scores_np, sb),

                    "truncated_best_pre_idx": tb,
                    "truncated_best_iou": float(trunc_best_iou[gi]),
                    "truncated_best_score":
                        score_value(pre_scores_np, tb),

                    "failure_type": "PENDING",
                    "ambiguity_reason": "",
                })

                for prefix, boxes_np, idx in [
                    ("pre_best", pre_boxes_np, pb),
                    ("pre_match", pre_boxes_np, pmb),
                    ("post_best", post_boxes_np, qb),
                    ("post_match", post_boxes_np, qmb),
                    ("suppressed", pre_boxes_np, sb),
                    ("truncated", pre_boxes_np, tb),
                ]:
                    b = box_values(boxes_np, idx)
                    row[prefix + "_x1"] = b[0]
                    row[prefix + "_y1"] = b[1]
                    row[prefix + "_x2"] = b[2]
                    row[prefix + "_y2"] = b[3]

                # Priority before F3/F5 confidence split.
                if (
                    row["suppressed_best_iou"] >= 0.5
                    and row["post_best_iou"] < 0.5
                ):
                    row["failure_type"] = "F4"

                elif (
                    row["pre_best_iou"] < 0.1
                    and row["post_best_iou"] < 0.1
                ):
                    row["failure_type"] = "F1"

                elif (
                    max(
                        row["pre_best_iou"],
                        row["post_best_iou"]
                    ) < 0.5
                    and max(
                        row["pre_best_iou"],
                        row["post_best_iou"]
                    ) >= 0.1
                ):
                    if max(
                        row["pre_match_iou"],
                        row["post_match_iou"]
                    ) >= 0.1:
                        row["failure_type"] = "F2"
                    else:
                        row["failure_type"] = "F-ambiguous"
                        row["ambiguity_reason"] = (
                            "one_to_one_conflict_localization"
                        )

                elif row["post_best_iou"] >= 0.5:
                    if row["post_match_iou"] >= 0.5:
                        row["failure_type"] = "SUCCESS_PENDING_Q1"
                    else:
                        row["failure_type"] = "F-ambiguous"
                        row["ambiguity_reason"] = (
                            "one_to_one_conflict_success"
                        )

                elif (
                    row["pre_best_iou"] >= 0.5
                    and row["post_best_iou"] < 0.5
                ):
                    row["failure_type"] = "F-ambiguous"

                    if row["truncated_best_iou"] >= 0.5:
                        row["ambiguity_reason"] = (
                            "max_per_img_truncation"
                        )
                    else:
                        row["ambiguity_reason"] = (
                            "high_iou_pre_disappeared_not_confirmed_nms"
                        )

                else:
                    row["failure_type"] = "F-ambiguous"
                    row["ambiguity_reason"] = "unresolved"

                records.append(row)

        done = min(st + batch_size, len(images))
        if done % 100 < batch_size or done == len(images):
            print(
                f"PROGRESS {done}/{len(images)} "
                f"targets={len(records)}",
                flush=True
            )

    handle.remove()

    if reconstruction_mismatches != 0:
        raise RuntimeError(
            "Ppre/Ppost reconstruction mismatch count="
            + str(reconstruction_mismatches)
        )

    # ========================================================
    # F3 / F5 confidence quartiles
    # ========================================================
    groups = defaultdict(list)

    for r in records:
        if r["failure_type"] == "SUCCESS_PENDING_Q1":
            groups[
                (r["category_id"], r["scale_group"])
            ].append(float(r["post_match_score"]))

    q1 = {}
    for key, vals in groups.items():
        q1[key] = float(np.quantile(np.asarray(vals), 0.25))

    for r in records:
        if r["failure_type"] != "SUCCESS_PENDING_Q1":
            continue

        key = (r["category_id"], r["scale_group"])
        threshold = q1[key]
        r["confidence_q1"] = threshold

        if float(r["post_match_score"]) <= threshold:
            r["failure_type"] = "F3"
        else:
            r["failure_type"] = "F5"

    for r in records:
        if "confidence_q1" not in r:
            key = (r["category_id"], r["scale_group"])
            r["confidence_q1"] = (
                float(q1[key]) if key in q1 else float("nan")
            )

    valid_labels = {
        "F1", "F2", "F3", "F4", "F5", "F-ambiguous"
    }

    bad = [
        r for r in records
        if r["failure_type"] not in valid_labels
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} target instances lack final label"
        )

    # ========================================================
    # Core artifacts
    # ========================================================
    table = pa.Table.from_pylist(records)

    pq.write_table(
        table,
        out / "failure_instances.parquet",
        compression="zstd"
    )

    # Readability gate.
    reloaded = pq.read_table(out / "failure_instances.parquet")
    if reloaded.num_rows != len(records):
        raise RuntimeError("Parquet row-count mismatch")

    write_atlas_csv(records, out / "error_atlas.csv")

    with open(out / "confidence_q1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category_id", "category_name",
                "scale_group", "n_success_candidates",
                "confidence_q1"
            ]
        )
        w.writeheader()

        for (cat, scale_group), threshold in sorted(q1.items()):
            w.writerow({
                "category_id": cat,
                "category_name": CATEGORIES[cat],
                "scale_group": scale_group,
                "n_success_candidates": len(groups[(cat, scale_group)]),
                "confidence_q1": threshold,
            })

    ambiguity_counts = Counter(
        r["ambiguity_reason"]
        for r in records
        if r["failure_type"] == "F-ambiguous"
    )

    with open(
        out / "ambiguity_summary.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["ambiguity_reason", "count"])
        for k, v in sorted(ambiguity_counts.items()):
            w.writerow([k, v])

    ambiguous = [
        r for r in records
        if r["failure_type"] == "F-ambiguous"
    ]
    rng = random.Random(0)
    audit = (
        ambiguous
        if len(ambiguous) <= 500
        else rng.sample(ambiguous, 500)
    )

    audit_fields = [
        "image_id", "file_name", "line_no",
        "category_id", "category_name",
        "scale_group", "gt_x1", "gt_y1", "gt_x2", "gt_y2",
        "pre_best_iou", "post_best_iou",
        "pre_match_iou", "post_match_iou",
        "suppressed_best_iou", "truncated_best_iou",
        "pre_best_survival", "ambiguity_reason"
    ]

    with open(
        out / "ambiguous_audit.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=audit_fields)
        w.writeheader()
        for r in audit:
            w.writerow({k: r[k] for k in audit_fields})

    # ========================================================
    # Frozen image lists
    # ========================================================
    image_counts = defaultdict(Counter)

    for r in records:
        image_counts[r["file_name"]][r["failure_type"]] += 1

    failure_images = []
    retention_images = []

    for fn, c in sorted(image_counts.items()):
        failure_n = sum(c[k] for k in ["F1", "F2", "F3", "F4"])

        if failure_n > 0:
            failure_images.append(fn)

        if c["F5"] > 0 and c["F5"] > failure_n:
            retention_images.append(fn)

    with open(out / "failure_images.txt", "w", encoding="utf-8") as f:
        for x in failure_images:
            f.write(x + "\n")

    with open(out / "retention_images.txt", "w", encoding="utf-8") as f:
        for x in retention_images:
            f.write(x + "\n")

    # ========================================================
    # Typical case figures
    # ========================================================
    draw_cases(
        records,
        Path(args.train_root) / "images",
        out / "typical_cases"
    )

    # ========================================================
    # Summaries
    # ========================================================
    counts = Counter(r["failure_type"] for r in records)
    total_targets = len(records)

    overlap = len(
        set(failure_images).intersection(retention_images)
    )

    meta = {
        "experiment_id": "E00-ATLAS-01",
        "processed_images": len(images),
        "total_valid_detection_gt": total_valid_gt,
        "target_instances": total_targets,
        "prediction_counts": {
            "Ppre_total": total_pre,
            "Ppost_total": total_post,
        },
        "prepost_reconstruction": {
            "mismatch_images": reconstruction_mismatches,
            "max_box_abs_diff": max_box_diff,
            "max_score_abs_diff": max_score_diff,
            "label_mismatch_total": label_mismatch_total,
        },
        "test_cfg": {
            "score_thr": float(test_cfg.score_thr),
            "nms_pre": int(test_cfg.nms_pre),
            "nms": dict(test_cfg.nms),
            "max_per_img": int(test_cfg.max_per_img),
        },
        "failure_counts": {
            k: int(counts[k])
            for k in [
                "F1", "F2", "F3", "F4",
                "F5", "F-ambiguous"
            ]
        },
        "failure_rates": {
            k: float(counts[k] / total_targets)
            if total_targets else 0.0
            for k in [
                "F1", "F2", "F3", "F4",
                "F5", "F-ambiguous"
            ]
        },
        "ambiguity_reasons": dict(ambiguity_counts),
        "confidence_q1": {
            f"{cat}:{scale}": value
            for (cat, scale), value in sorted(q1.items())
        },
        "image_lists": {
            "failure_images": len(failure_images),
            "retention_images": len(retention_images),
            "overlap": overlap,
        },
        "dataset_manifest_sha256": args.dataset_sha,
        "checkpoint_sha256": args.checkpoint_sha,
        "config_sha256": args.config_sha,
        "wall_time_seconds": time.time() - start,
    }

    with open(
        out / "prediction_index_meta.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "index_semantics": {
                    "pre_*_idx":
                        "index into with_nms=False Ppre for that image",
                    "post_*_idx":
                        "index into frozen normal Ppost for that image",
                    "post_match_source_pre_idx":
                        "Ppre source index of one-to-one matched Ppost",
                    "suppressed_best_pre_idx":
                        "best correct-class Ppre candidate removed by NMS",
                    "truncated_best_pre_idx":
                        "best correct-class NMS-kept candidate removed only by max_per_img",
                },
                "prepost_reconstruction": meta[
                    "prepost_reconstruction"
                ],
                "test_cfg": meta["test_cfg"],
            },
            f, indent=2
        )

    with open(out / "atlas_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("E00_BUILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
