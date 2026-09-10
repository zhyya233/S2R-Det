from __future__ import annotations

from typing import Dict, Iterable, List

import torch.nn as nn

from s2r_det.adapters.conv_adapter import (
    freeze_base_train_conv_adapters,
    insert_residual_parallel_conv_adapters,
)
from s2r_det.adapters.flcr import (
    freeze_base_train_adapters,
    insert_flcr,
)


NORM_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def freeze_bn_running_stats(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def configure_variant(
    model,
    variant: str,
    controls: Dict,
):
    """Configure only the parameter/update scope.

    Data sampler and retention loss are intentionally orthogonal and are
    configured by the training protocol.
    """

    variant = variant.lower()

    for p in model.parameters():
        p.requires_grad = False

    meta = {
        "variant": variant,
        "adapter_type": "none",
        "insertion_layers": [],
        "rank": None,
        "bn_running_stats_frozen": True,
    }

    if variant == "frozen":
        pass

    elif variant == "head_only":
        for p in model.bbox_head.parameters():
            p.requires_grad = True

    elif variant == "norm_bias":
        norm_param_ids = set()

        for m in model.modules():
            if isinstance(m, NORM_TYPES):
                for p in m.parameters(recurse=False):
                    norm_param_ids.add(id(p))

        for name, p in model.named_parameters():
            if (
                id(p) in norm_param_ids
                or name.endswith(".bias")
            ):
                p.requires_grad = True

    elif variant == "full_ft":
        for p in model.parameters():
            p.requires_grad = True

        meta["bn_running_stats_frozen"] = False

    elif variant == "conv_adapter":
        layers = insert_residual_parallel_conv_adapters(
            model,
            gamma=int(
                controls["conv_adapter"]["gamma"]
            ),
        )

        freeze_base_train_conv_adapters(model)

        meta.update({
            "adapter_type": "Conv-Adapter Residual Parallel",
            "insertion_layers": layers,
            "rank": None,
        })

    elif variant == "unguided_flcr":
        layers = list(
            controls["p00_05_unguided_flcr"]["layers"]
        )

        insert_flcr(
            model,
            layers,
            rank=8,
        )

        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": 8,
        })

    elif variant == "s2r_initial":
        layers = list(
            controls["p00_06_s2r_initial"]["layers"]
        )

        insert_flcr(
            model,
            layers,
            rank=8,
        )

        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": 8,
        })

    elif variant == "r00_location":
        if controls.get("stage") != "R00" or int(controls.get("round", -1)) not in (1, 2):
            raise ValueError("r00_location requires R00 Round-1 manifest")

        layers = list(controls["layers"])
        rank = int(controls["rank"])

        insert_flcr(model, layers, rank=rank)
        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": rank,
            "experiment_id": controls.get("experiment_id"),
            "control": controls.get("control"),
        })

    elif variant == "m00_head_r8":
        if controls.get("stage") not in {"M00", "K00", "G00"}:
            raise ValueError(
                "m00_head_r8 requires M00, K00 or G00 controls"
            )

        expected_layers = [
            "bbox_head.cls_convs.0.1.conv",
            "bbox_head.reg_convs.0.1.conv",
        ]

        layers = list(controls.get("layers", expected_layers))
        rank = int(controls.get("rank", 8))

        if layers != expected_layers or rank != 8:
            raise ValueError("m00_head_r8 is frozen to Head rank8")

        insert_flcr(model, layers, rank=rank)
        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": rank,
            "experiment_id": controls.get("experiment_id"),
            "control": controls.get("control", "head"),
        })


    elif variant == "a00_location":
        if controls.get("stage") != "A00":
            raise ValueError(
                "a00_location requires A00 controls"
            )

        layers = list(controls["layers"])
        rank = int(controls.get("rank", 8))

        insert_flcr(
            model,
            layers,
            rank=rank,
        )
        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": rank,
            "experiment_id": controls.get("experiment_id"),
            "control": controls.get("control"),
            "a00_family": controls.get("a00_family"),
        })

    elif variant == "a00_u2":
        if controls.get("stage") != "A00":
            raise ValueError(
                "a00_u2 requires A00 controls"
            )

        expected_layers = [
            "bbox_head.cls_convs.0.1.conv",
            "bbox_head.reg_convs.0.1.conv",
        ]

        layers = list(controls.get("layers", expected_layers))
        rank = int(controls.get("rank", 8))

        if layers != expected_layers or rank != 8:
            raise ValueError(
                "A00 U2 is frozen to Head rank8"
            )

        insert_flcr(
            model,
            layers,
            rank=rank,
        )
        freeze_base_train_adapters(model)

        bn_affine_names = []

        for module_name, module in model.named_modules():
            if not isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                continue

            for local_name, p in module.named_parameters(
                recurse=False
            ):
                p.requires_grad_(True)
                bn_affine_names.append(
                    (
                        module_name + "." + local_name
                        if module_name
                        else local_name
                    )
                )

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": rank,
            "update_scope": "U2_FLCR_PLUS_BN_AFFINE",
            "bn_affine_names": bn_affine_names,
            "bn_running_stats_frozen": True,
            "experiment_id": controls.get("experiment_id"),
            "a00_family": "A00-08",
        })

    elif variant == "a00_u3":
        if controls.get("stage") != "A00":
            raise ValueError(
                "a00_u3 requires A00 controls"
            )

        expected_layers = [
            "bbox_head.cls_convs.0.1.conv",
            "bbox_head.reg_convs.0.1.conv",
        ]

        layers = list(controls.get("layers", expected_layers))
        rank = int(controls.get("rank", 8))

        if layers != expected_layers or rank != 8:
            raise ValueError(
                "A00 U3 is frozen to Head rank8"
            )

        insert_flcr(
            model,
            layers,
            rank=rank,
        )
        freeze_base_train_adapters(model)

        for p in model.bbox_head.parameters():
            p.requires_grad_(True)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": rank,
            "update_scope": "U3_FLCR_PLUS_DETECTION_HEAD",
            "bn_running_stats_frozen": True,
            "experiment_id": controls.get("experiment_id"),
            "a00_family": "A00-08",
        })

    elif variant == "a00_hf_fixed":
        if controls.get("stage") != "A00":
            raise ValueError(
                "a00_hf_fixed requires A00 controls"
            )

        expected_layers = [
            "bbox_head.cls_convs.0.1.conv",
            "bbox_head.reg_convs.0.1.conv",
        ]

        layers = list(controls.get("layers", expected_layers))
        rank = int(controls.get("rank", 8))

        if layers != expected_layers or rank != 8:
            raise ValueError(
                "A00 HF-fixed is frozen to Head rank8"
            )

        insert_flcr(
            model,
            layers,
            rank=rank,
            fixed_depthwise_kernel="laplacian4",
        )
        freeze_base_train_adapters(model)

        meta.update({
            "adapter_type": "FLCR-HF-FIXED",
            "insertion_layers": layers,
            "rank": rank,
            "fixed_depthwise_kernel": "laplacian4",
            "hf_status": "EXPLORATORY",
            "experiment_id": controls.get("experiment_id"),
            "a00_family": "A00-09",
        })

    elif variant == "s2r_u2":
        # U2: FCSL Top-2 FLCR + BatchNorm affine only.
        # Running mean/variance remain frozen by the training loop
        # through bn_running_stats_frozen=True.
        layers = list(
            controls["p00_06_s2r_initial"]["layers"]
        )

        insert_flcr(
            model,
            layers,
            rank=8,
        )

        freeze_base_train_adapters(model)

        bn_affine_names = []

        for module_name, module in model.named_modules():
            if not isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                continue

            for local_name, p in module.named_parameters(
                recurse=False
            ):
                p.requires_grad_(True)
                bn_affine_names.append(
                    (
                        module_name + "." + local_name
                        if module_name
                        else local_name
                    )
                )

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": 8,
            "update_scope": "U2_FLCR_PLUS_BN_AFFINE",
            "bn_affine_names": bn_affine_names,
            "bn_running_stats_frozen": True,
        })

    elif variant == "s2r_u3":
        # U3: FCSL Top-2 FLCR + full detection head.
        # Backbone/neck remain frozen apart from the approved FLCRs.
        layers = list(
            controls["p00_06_s2r_initial"]["layers"]
        )

        insert_flcr(
            model,
            layers,
            rank=8,
        )

        freeze_base_train_adapters(model)

        for p in model.bbox_head.parameters():
            p.requires_grad_(True)

        meta.update({
            "adapter_type": "FLCR",
            "insertion_layers": layers,
            "rank": 8,
            "update_scope": "U3_FLCR_PLUS_DETECTION_HEAD",
            "bn_running_stats_frozen": True,
        })

    else:
        raise ValueError(
            f"unknown P00 variant: {variant}"
        )

    trainable = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    meta.update({
        "trainable_names": trainable,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_ratio": (
            trainable_params / total_params
            if total_params
            else 0.0
        ),
    })

    return meta
