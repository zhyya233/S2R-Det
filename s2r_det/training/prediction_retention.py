from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from mmdet.structures.bbox import distance2bbox

try:
    from mmdet.models.utils import filter_scores_and_topk
except ImportError:
    from mmdet.models.utils.misc import filter_scores_and_topk


def flatten_dense_predictions(
    head,
    cls_scores,
    bbox_preds,
    sample,
):
    """Exact dense-location ordering used by S00 FCSL."""
    metas = [dict(sample.metainfo)]
    featmap_sizes = [
        x.shape[-2:]
        for x in cls_scores
    ]

    device = cls_scores[0].device

    anchor_list, _ = head.get_anchors(
        featmap_sizes,
        metas,
        device=device,
    )

    flatten_cls = torch.cat(
        [
            x.permute(
                0, 2, 3, 1
            ).reshape(
                1,
                -1,
                head.cls_out_channels,
            )
            for x in cls_scores
        ],
        dim=1,
    )[0]

    decoded = []

    for anchor, pred in zip(
        anchor_list[0],
        bbox_preds,
    ):
        a = anchor.reshape(-1, 4)

        pp = pred.permute(
            0, 2, 3, 1
        ).reshape(
            1,
            -1,
            4,
        )

        decoded.append(
            distance2bbox(
                a,
                pp,
            )[0]
        )

    flatten_bbox = torch.cat(
        decoded,
        dim=0,
    )

    return flatten_cls, flatten_bbox


def reconstruct_ppre_dense_map(
    head,
    cls_scores,
    bbox_preds,
    img_meta,
    test_cfg,
    rescale: bool = True,
):
    """Exact mapping from official with_nms=False Ppre to dense locations.

    Reproduces the relevant MMDetection 3.3.0 sequence:

      _predict_by_feat_single:
        activation
        -> score_thr
        -> per-level nms_pre top-k
        -> bbox decode

      _bbox_post_process(with_nms=False):
        optional rescale
        -> min_bbox_size filtering

    No NMS is applied.

    Returns:
      candidate_scores: [Npre]
      candidate_labels: [Npre]
      candidate_dense_indices: [Npre]
    """

    cfg = test_cfg

    score_thr = float(
        cfg.get(
            "score_thr",
            0.0,
        )
    )

    nms_pre = int(
        cfg.get(
            "nms_pre",
            -1,
        )
    )

    featmap_sizes = [
        x.shape[-2:]
        for x in cls_scores
    ]

    mlvl_priors = (
        head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
        )
    )

    all_scores = []
    all_labels = []
    all_bbox_preds = []
    all_priors = []
    all_dense = []

    offset = 0

    for (
        cls_score4,
        bbox_pred4,
        priors,
    ) in zip(
        cls_scores,
        bbox_preds,
        mlvl_priors,
    ):
        # predict_by_feat(select_single_mlvl) removes batch dimension.
        cls_score = cls_score4[0]

        bbox_pred = bbox_pred4[0]

        dim = head.bbox_coder.encode_size

        bbox_pred = (
            bbox_pred
            .permute(1, 2, 0)
            .reshape(-1, dim)
        )

        cls_logits = (
            cls_score
            .permute(1, 2, 0)
            .reshape(
                -1,
                head.cls_out_channels,
            )
        )

        if getattr(
            head.loss_cls,
            "custom_cls_channels",
            False,
        ):
            scores = (
                head.loss_cls
                .get_activation(
                    cls_logits
                )
            )
        elif head.use_sigmoid_cls:
            scores = (
                cls_logits.sigmoid()
            )
        else:
            scores = (
                cls_logits
                .softmax(-1)[:, :-1]
            )

        dense_source = (
            torch.arange(
                cls_logits.shape[0],
                device=cls_logits.device,
                dtype=torch.long,
            )
            + offset
        )

        (
            kept_scores,
            labels,
            keep_idxs,
            filtered,
        ) = filter_scores_and_topk(
            scores,
            score_thr,
            nms_pre,
            dict(
                bbox_pred=bbox_pred,
                priors=priors,
                dense_source=dense_source,
            ),
        )

        all_scores.append(
            kept_scores
        )

        all_labels.append(
            labels
        )

        all_bbox_preds.append(
            filtered["bbox_pred"]
        )

        all_priors.append(
            filtered["priors"]
        )

        all_dense.append(
            filtered["dense_source"]
        )

        offset += cls_logits.shape[0]

    candidate_scores = torch.cat(
        all_scores
    )

    candidate_labels = torch.cat(
        all_labels
    )

    candidate_bbox_pred = torch.cat(
        all_bbox_preds
    )

    candidate_priors = torch.cat(
        all_priors
    )

    candidate_dense = torch.cat(
        all_dense
    )

    # Same decode as _predict_by_feat_single.
    candidate_boxes = (
        head.bbox_coder.decode(
            candidate_priors,
            candidate_bbox_pred,
            max_shape=img_meta["img_shape"],
        )
    )

    # Same rescale ordering as _bbox_post_process.
    if rescale:
        sf = img_meta.get(
            "scale_factor",
            None,
        )

        if sf is None:
            raise RuntimeError(
                "scale_factor missing for rescale=True"
            )

        scale = (
            candidate_boxes.new_tensor(
                [
                    1.0 / float(sf[0]),
                    1.0 / float(sf[1]),
                    1.0 / float(sf[0]),
                    1.0 / float(sf[1]),
                ]
            )
        )

        candidate_boxes = (
            candidate_boxes
            * scale
        )

    # Crucial MMDetection postprocess step that still executes when
    # with_nms=False.
    min_bbox_size = float(
        cfg.get(
            "min_bbox_size",
            -1,
        )
    )

    if min_bbox_size >= 0:
        w = (
            candidate_boxes[:, 2]
            - candidate_boxes[:, 0]
        )

        h = (
            candidate_boxes[:, 3]
            - candidate_boxes[:, 1]
        )

        valid = (
            (w > min_bbox_size)
            & (h > min_bbox_size)
        )

        candidate_scores = (
            candidate_scores[valid]
        )

        candidate_labels = (
            candidate_labels[valid]
        )

        candidate_dense = (
            candidate_dense[valid]
        )

    return (
        candidate_scores,
        candidate_labels,
        candidate_dense,
    )


def pairwise_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    if (
        boxes1.numel() == 0
        or boxes2.numel() == 0
    ):
        return boxes1.new_zeros(
            (
                boxes1.shape[0],
                boxes2.shape[0],
            )
        )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (
        rb - lt
    ).clamp(min=0)

    inter = (
        wh[..., 0]
        * wh[..., 1]
    )

    area1 = (
        (
            boxes1[:, 2]
            - boxes1[:, 0]
        ).clamp(min=0)
        *
        (
            boxes1[:, 3]
            - boxes1[:, 1]
        ).clamp(min=0)
    )

    area2 = (
        (
            boxes2[:, 2]
            - boxes2[:, 0]
        ).clamp(min=0)
        *
        (
            boxes2[:, 3]
            - boxes2[:, 1]
        ).clamp(min=0)
    )

    return inter / (
        area1[:, None]
        + area2[None, :]
        - inter
        + 1e-12
    )


def greedy_f5_match_state(
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thr: float = 0.5,
) -> torch.Tensor:
    """Class-consistent IoU-priority one-to-one match state."""

    k = candidate_boxes.shape[0]

    state = torch.zeros(
        k,
        dtype=torch.uint8,
        device=candidate_boxes.device,
    )

    if (
        k == 0
        or gt_boxes.numel() == 0
    ):
        return state

    iou = pairwise_iou(
        candidate_boxes,
        gt_boxes,
    )

    class_ok = (
        candidate_labels[:, None]
        == gt_labels[None, :]
    )

    valid = (
        class_ok
        & (iou >= iou_thr)
    )

    pairs = torch.nonzero(
        valid,
        as_tuple=False,
    )

    if pairs.numel() == 0:
        return state

    vals = iou[
        pairs[:, 0],
        pairs[:, 1],
    ]

    # Stable descending IoU. Pair indices provide deterministic tie order.
    order = torch.argsort(
        vals,
        descending=True,
        stable=True,
    )

    used_c = set()
    used_g = set()

    for oi in order.detach().cpu().tolist():
        ci = int(
            pairs[oi, 0].item()
        )

        gi = int(
            pairs[oi, 1].item()
        )

        if (
            ci in used_c
            or gi in used_g
        ):
            continue

        used_c.add(ci)
        used_g.add(gi)
        state[ci] = 1

    return state


def bernoulli_kl_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL for RTMDet's independent sigmoid class logits.

    Computes KL(Bernoulli(p_teacher) || Bernoulli(p_student)) per class,
    with temperature scaling and the conventional T^2 factor.
    """

    t = float(temperature)

    teacher_scaled = (
        teacher_logits / t
    )

    student_scaled = (
        student_logits / t
    )

    with torch.no_grad():
        p_teacher = torch.sigmoid(
            teacher_scaled
        )

    cross_entropy = (
        F.binary_cross_entropy_with_logits(
            student_scaled,
            p_teacher,
            reduction="none",
        )
    )

    teacher_entropy_cross = (
        F.binary_cross_entropy_with_logits(
            teacher_scaled,
            p_teacher,
            reduction="none",
        )
    )

    kl = (
        cross_entropy
        - teacher_entropy_cross
    )

    return (
        kl.mean()
        * (t * t)
    )


def prediction_retention_loss(
    student_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
    gt_match: torch.Tensor,
    temperature: float = 2.0,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Prediction Retention component, without lambda_ret.

    Classification:
      Bernoulli KL over all cached top-K candidates.

    Box:
      SmoothL1 only on cached candidates matched one-to-one to F5 GT.
      beta is SmoothL1 beta and the route coefficient is also fixed to 1.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits shape mismatch"
        )

    if student_boxes.shape != teacher_boxes.shape:
        raise ValueError(
            "student/teacher boxes shape mismatch"
        )

    if gt_match.ndim != 1:
        raise ValueError(
            "gt_match must be 1-D"
        )

    cls_loss = bernoulli_kl_logits(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )

    mask = gt_match.to(
        dtype=torch.bool,
        device=student_boxes.device,
    )

    if bool(mask.any()):
        box_loss = F.smooth_l1_loss(
            student_boxes[mask],
            teacher_boxes[mask],
            beta=float(beta),
            reduction="mean",
        )
    else:
        box_loss = (
            student_boxes.sum()
            * 0.0
        )

    total = (
        cls_loss
        + box_loss
    )

    return {
        "loss_ret_cls": cls_loss,
        "loss_ret_box": box_loss,
        "loss_ret": total,
    }


def flatten_dense_predictions_batch(
    head,
    cls_scores,
    bbox_preds,
    data_samples,
):
    """Differentiable batch dense RTMDet logits and decoded boxes.

    Output:
      logits: [B, N, C]
      boxes:  [B, N, 4]

    Ordering is identical to the frozen single-image FCSL/cache dense order.
    """

    batch_size = cls_scores[0].shape[0]

    if batch_size != len(data_samples):
        raise ValueError(
            "batch/data_samples size mismatch"
        )

    metas = [
        dict(sample.metainfo)
        for sample in data_samples
    ]

    featmap_sizes = [
        x.shape[-2:]
        for x in cls_scores
    ]

    device = cls_scores[0].device

    anchor_list, _ = head.get_anchors(
        featmap_sizes,
        metas,
        device=device,
    )

    flatten_cls = torch.cat(
        [
            x.permute(
                0, 2, 3, 1
            ).reshape(
                batch_size,
                -1,
                head.cls_out_channels,
            )
            for x in cls_scores
        ],
        dim=1,
    )

    per_image_boxes = []

    for b in range(batch_size):
        decoded = []

        for anchor, pred in zip(
            anchor_list[b],
            bbox_preds,
        ):
            a = anchor.reshape(
                -1, 4
            )

            pp = (
                pred[b]
                .permute(1, 2, 0)
                .reshape(-1, 4)
            )

            decoded.append(
                distance2bbox(
                    a,
                    pp,
                )
            )

        per_image_boxes.append(
            torch.cat(
                decoded,
                dim=0,
            )
        )

    flatten_bbox = torch.stack(
        per_image_boxes,
        dim=0,
    )

    return (
        flatten_cls,
        flatten_bbox,
    )
