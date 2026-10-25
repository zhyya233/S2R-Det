"""Detached teacher Bernoulli KL; independently masked pooled-entry box loss."""

import torch
import torch.nn.functional as F


def bernoulli_kl_logits(student_logits, teacher_logits, temperature=2.0):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2 or temperature <= 0:
        raise ValueError("invalid retention logits/temperature")
    if student_logits.numel() == 0:
        return student_logits.sum() * 0
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("non-finite retention logits")
    s = student_logits / temperature
    t = teacher_logits.detach() / temperature
    p = t.sigmoid()
    return (
        F.binary_cross_entropy_with_logits(s, p, reduction="none")
        - F.binary_cross_entropy_with_logits(t, p, reduction="none")
    ).mean() * temperature**2


def prediction_retention_loss(
    student_logits,
    student_boxes,
    teacher_logits,
    teacher_boxes,
    box_mask,
    temperature=2.0,
    beta=1.0,
    delta=1.0,
):
    if (
        student_boxes.shape != teacher_boxes.shape
        or student_boxes.shape != (student_logits.shape[0], 4)
        or box_mask.shape != (student_logits.shape[0],)
    ):
        raise ValueError("invalid retention shapes")
    if delta <= 0 or beta < 0:
        raise ValueError("invalid box transition/weight")
    if not torch.isfinite(student_boxes).all() or not torch.isfinite(teacher_boxes).all():
        raise ValueError("non-finite retention boxes")
    cls = bernoulli_kl_logits(student_logits, teacher_logits, temperature)
    mask = box_mask.to(device=student_boxes.device, dtype=torch.bool)
    box = (
        F.smooth_l1_loss(
            student_boxes[mask], teacher_boxes.detach()[mask], beta=delta, reduction="mean"
        )
        if mask.any()
        else student_boxes.sum() * 0
    )
    return {"classification": cls, "box": box, "total": cls + beta * box}


def pooled_retention_loss(pairs, zero, temperature=2.0, beta=1.0, delta=1.0):
    """pairs contain aligned tensors for all eligible images in one minibatch."""
    if not pairs:
        return {"classification": zero * 0, "box": zero * 0, "total": zero * 0}
    columns = [torch.cat([p[i] for p in pairs], dim=0) for i in range(5)]
    return prediction_retention_loss(*columns, temperature=temperature, beta=beta, delta=delta)
