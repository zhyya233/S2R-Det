"""Raw valid candidate-class entries, not score-filtered/NMS-selected entries."""

import torch
from s2r_det.evaluation.matching import greedy_match


def build_cache_entry(image_id, logits, boxes, f5_gt, k=100):
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4) or k < 1:
        raise ValueError("invalid dense cache inputs")
    logits = logits.detach()
    boxes = boxes.detach()
    valid = (
        torch.isfinite(logits).all(1)
        & torch.isfinite(boxes).all(1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    candidates = []
    scores = logits.sigmoid().cpu()
    for dense in torch.where(valid)[0].tolist():
        for cls in range(logits.shape[1]):
            candidates.append((-float(scores[dense, cls]), dense, cls))
    chosen = sorted(candidates)[:k]
    indices = [i for _, i, _ in chosen]
    classes = [c for _, _, c in chosen]
    selected_boxes = boxes[indices].cpu().tolist()
    predicted = [{"category_id": c, "box": b} for c, b in zip(classes, selected_boxes)]
    matched = greedy_match(f5_gt, predicted, 0.5)
    gt_by_entry = {p: str(f5_gt[g]["annotation_id"]) for g, p in matched.items()}
    return {
        "image_id": str(image_id),
        "dense_indices": indices,
        "candidate_classes": classes,
        "teacher_logits": logits[indices].cpu().tolist(),
        "teacher_boxes": selected_boxes,
        "box_mask": [i in gt_by_entry for i in range(len(chosen))],
        "matched_gt_ids": [gt_by_entry.get(i) for i in range(len(chosen))],
        "num_classes": logits.shape[1],
        "dense_count": logits.shape[0],
    }
