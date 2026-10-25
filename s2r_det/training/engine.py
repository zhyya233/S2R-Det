"""Explicit final-budget repair. No hidden inference or checkpoint selection."""

from __future__ import annotations
import hashlib
import torch
from s2r_det.retention.losses import pooled_retention_loss
from s2r_det.training.protocol import (
    learning_rate_factor,
    optimizer_groups,
    freeze_running_stats,
    seed_everything,
)
from s2r_det.utils.io import require_identity


def frozen_fingerprints(model):
    return {
        n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }


def repair(adapter, dataset, batches, protocol, cache=None, cache_identity=None):
    seed_everything(protocol.seed)
    model = adapter.model
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("no declared trainable parameters")
    if protocol.retention_lambda > 0:
        if cache is None or cache_identity is None:
            raise ValueError("retention requires a validated fixed-teacher cache")
        require_identity(cache["identity"], cache_identity)
    before = frozen_fingerprints(model)
    running = {
        n: b.detach().clone()
        for n, b in model.named_buffers()
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, protocol.weight_decay),
        lr=protocol.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    completed = 0
    last = None
    model.train()
    freeze_running_stats(model)
    prepared_batches = (
        adapter.iter_prepared_batches(dataset, batches, protocol.seed, protocol.workers)
        if hasattr(adapter, "iter_prepared_batches")
        else ((batch, adapter.prepare(dataset, batch["image_ids"])) for batch in batches)
    )
    for batch, prepared in prepared_batches:
        if completed >= protocol.steps:
            raise ValueError("sampler supplied more than prescribed updates")
        if batch["step"] != completed or len(batch["image_ids"]) != protocol.batch_size:
            raise ValueError("sampler step/batch-size mismatch")
        rate = protocol.learning_rate * learning_rate_factor(
            completed, protocol.steps, protocol.warmup_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        native = adapter.detection_loss(prepared)
        terms = []
        for name, value in native.items():
            if "loss" in name:
                terms.extend(value if isinstance(value, (list, tuple)) else [value])
        if not terms:
            raise ValueError("native detector returned no loss components")
        detection = sum(x.mean() for x in terms)
        retention = detection * 0
        if protocol.retention_lambda > 0:
            logits, boxes, _ = adapter.dense(prepared)
            pairs = []
            for b, image_id in enumerate(batch["image_ids"]):
                entry = cache["entries"].get(str(image_id))
                eligible = str(image_id) in cache["eligible_image_ids"]
                if eligible and entry is None:
                    raise ValueError("eligible image has no teacher cache")
                if not eligible:
                    continue
                if (
                    entry["dense_count"] != logits.shape[1]
                    or entry["num_classes"] != logits.shape[2]
                ):
                    raise ValueError("dense cache correspondence mismatch")
                ids = torch.tensor(entry["dense_indices"], device=logits.device, dtype=torch.long)
                if ids.numel() and (ids.min() < 0 or ids.max() >= logits.shape[1]):
                    raise ValueError("cache contains an invalid dense index")
                count = len(ids)
                classes = logits.shape[2]
                pairs.append(
                    (
                        logits[b, ids],
                        boxes[b, ids],
                        logits.new_tensor(entry["teacher_logits"]).reshape(count, classes),
                        boxes.new_tensor(entry["teacher_boxes"]).reshape(count, 4),
                        torch.tensor(entry["box_mask"], device=boxes.device, dtype=torch.bool),
                    )
                )
            retention = pooled_retention_loss(pairs, detection)["total"]
        loss = detection + protocol.retention_lambda * retention
        if not torch.isfinite(loss):
            raise ValueError("non-finite optimization loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("non-finite optimization gradient")
        optimizer.step()
        completed += 1
        last = float(loss.detach())
    if completed != protocol.steps:
        raise ValueError("sampler ended before prescribed final update")
    if before != frozen_fingerprints(model):
        raise RuntimeError("frozen detector parameter changed")
    current = dict(model.named_buffers())
    if any(not torch.equal(b, current[n]) for n, b in running.items()):
        raise RuntimeError("frozen normalization running state changed")
    return {
        "completed_steps": completed,
        "final_loss": last,
        "checkpoint_selection": "final prescribed update; no test-based selection",
        "seed": protocol.seed,
        "frozen_state_verified": True,
    }
