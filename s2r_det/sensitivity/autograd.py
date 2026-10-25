"""Convolution outputs before normalization; shared calls pooled by elements."""

from __future__ import annotations
import math
import torch
from s2r_det.sensitivity.scoring import EPS


class ActivationRecorder:
    def __init__(self, model, paths):
        self.model = model
        self.paths = list(paths)
        self.calls = {n: [] for n in paths}
        self.handles = []

    def __enter__(self):
        modules = [self.model.get_submodule(name) for name in self.paths]
        if any(not isinstance(m, torch.nn.Conv2d) for m in modules):
            raise TypeError("record convolution output, not normalization/activation")
        if len({id(m) for m in modules}) != len(modules):
            raise ValueError("activation manifest must deduplicate shared aliases")
        for name, m in zip(self.paths, modules):

            def hook(module, args, output, n=name):
                self.calls[n].append(output)

            self.handles.append(m.register_forward_hook(hook))
        return self

    def clear(self):
        for values in self.calls.values():
            values.clear()

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def sensitivities(self, loss):
        if not torch.isfinite(loss).all():
            raise ValueError("non-finite diagnostic loss")
        tensors = [a for values in self.calls.values() for a in values if a.requires_grad]
        gradients = (
            torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
            if tensors
            else []
        )
        grads = {id(a): g for a, g in zip(tensors, gradients)}
        out = {}
        for name, values in self.calls.items():
            if not values:
                raise ValueError(f"eligible module not called: {name}")
            numerator = denom = count = 0
            for a in values:
                if a.dtype != torch.float32 or not torch.isfinite(a).all():
                    raise ValueError("FCSL requires finite FP32 activations")
                g = grads.get(id(a))
                count += a.numel()
                denom += float(a.detach().square().sum())
                if g is not None:
                    if not torch.isfinite(g).all():
                        raise ValueError("non-finite diagnostic gradient")
                    numerator += float((a * g).detach().abs().sum())
            out[name] = (numerator / count) / (math.sqrt(denom / count) + EPS)
        return out


def diagnostic_loss(head, logits, boxes, gt_box, class_index):
    if boxes.shape[0] == 0:
        raise ValueError("no dense candidate for diagnostic correspondence")
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise ValueError("non-finite dense prediction")
    with torch.no_grad():
        lt = torch.maximum(boxes[:, :2], gt_box[:2])
        rb = torch.minimum(boxes[:, 2:], gt_box[2:])
        inter = (rb - lt).clamp(min=0).prod(1)
        area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
        ga = (gt_box[2:] - gt_box[:2]).prod()
        index = int(torch.argmax(inter / (area + ga - inter).clamp(min=EPS)))
    label = torch.tensor([class_index], device=logits.device)
    quality = logits.new_ones(1)
    loss = head.loss_cls(
        logits[index : index + 1], (label, quality), weight=quality, avg_factor=1.0
    )
    loss = loss + head.loss_bbox(
        boxes[index : index + 1], gt_box[None], weight=quality, avg_factor=1.0
    )
    return loss, index
