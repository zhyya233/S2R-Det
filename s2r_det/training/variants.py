"""Named scientific update scopes; data selection/retention remain orthogonal."""

import torch
from s2r_det.adapters.flcr import insert_flcr, freeze_base_train_adapters
from s2r_det.adapters.conv_adapter import (
    insert_residual_parallel_conv_adapters,
    freeze_base_train_conv_adapters,
)
from s2r_det.training.protocol import freeze_running_stats


def configure_variant(model, name, placement=None, rank=8, norm_paths=None):
    model.requires_grad_(False)
    if name == "frozen":
        return []
    if name in {
        "flcr",
        "s2r_det",
        "unguided_flcr",
        "flcr_plus_norm",
        "flcr_plus_head",
        "flcr_fixed_spatial",
    }:
        if not placement:
            raise ValueError("explicit validated placement required")
        insert_flcr(model, placement, rank, "laplacian4" if name == "flcr_fixed_spatial" else None)
        freeze_base_train_adapters(model)
    elif name == "conv_adapter":
        insert_residual_parallel_conv_adapters(model, gamma=4)
        freeze_base_train_conv_adapters(model)
    elif name not in {"head_only", "norm_bias", "full_finetuning", "bottleneck_adapter"}:
        raise ValueError(
            f"unsupported method {name}; supply a documented external control factory when required"
        )
    if name in {"head_only", "flcr_plus_head"}:
        if not hasattr(model, "bbox_head"):
            raise ValueError("architecture must expose explicit head boundary")
        model.bbox_head.requires_grad_(True)
    if name in {"norm_bias", "flcr_plus_norm"}:
        if name == "flcr_plus_norm" and not norm_paths:
            raise ValueError(
                "FLCR+norm requires explicit normalization paths; the historical subset is not uniquely specified"
            )
        modules = (
            model.modules() if name == "norm_bias" else [model.get_submodule(n) for n in norm_paths]
        )
        for module in modules:
            if isinstance(
                module,
                (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm, torch.nn.LayerNorm),
            ):
                for p in module.parameters(recurse=False):
                    p.requires_grad_(True)
            elif name == "flcr_plus_norm":
                raise ValueError("norm_paths must identify normalization modules")
        if name == "norm_bias":
            for n, p in model.named_parameters():
                if n == "bias" or n.endswith(".bias"):
                    p.requires_grad_(True)
    if name == "full_finetuning":
        model.requires_grad_(True)
    if name == "bottleneck_adapter":
        raise ValueError(
            "YOLOv8n bottleneck control requires architecture-specific external factory; exact historical topology not specified"
        )
    freeze_running_stats(model)
    selected = [n for n, p in model.named_parameters() if p.requires_grad]
    if not selected:
        raise ValueError("no trainable parameters for selected method")
    return selected
