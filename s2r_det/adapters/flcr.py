import copy
from typing import Iterable, List

import torch
import torch.nn as nn


class FLCRConv2d(nn.Module):
    """
    Fusible Low-Rank Convolutional Repair.

    target:
        ordinary groups=1 3x3 Conv2d

    branch:
        V: 1x1 Cin -> rank
        D: rank-channel depthwise 3x3
        U: 1x1 rank -> Cout

    forward:
        target(x) + U(D(V(x)))

    U is zero initialized.
    """

    def __init__(self, base: nn.Conv2d, rank: int = 8, fixed_depthwise_kernel=None):
        super().__init__()

        if not isinstance(base, nn.Conv2d):
            raise TypeError(type(base))

        if tuple(base.kernel_size) != (3, 3):
            raise ValueError(
                f"FLCR requires 3x3 target, got {base.kernel_size}"
            )

        if int(base.groups) != 1:
            raise ValueError(
                f"FLCR requires groups=1, got {base.groups}"
            )

        self.base = base
        self.rank = int(rank)
        self.fixed_depthwise_kernel = fixed_depthwise_kernel

        self.repair_v = nn.Conv2d(
            base.in_channels,
            self.rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        self.repair_d = nn.Conv2d(
            self.rank,
            self.rank,
            kernel_size=3,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=self.rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        self.repair_u = nn.Conv2d(
            self.rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        nn.init.kaiming_uniform_(
            self.repair_v.weight,
            a=5 ** 0.5,
        )

        if self.fixed_depthwise_kernel is None:
            nn.init.kaiming_uniform_(
                self.repair_d.weight,
                a=5 ** 0.5,
            )
        elif self.fixed_depthwise_kernel == "laplacian4":
            kernel = torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0],
                ],
                device=base.weight.device,
                dtype=base.weight.dtype,
            ).reshape(1, 1, 3, 3)

            with torch.no_grad():
                self.repair_d.weight.copy_(
                    kernel.repeat(self.rank, 1, 1, 1)
                )

            self.repair_d.weight.requires_grad_(False)
        else:
            raise ValueError(
                "unknown fixed depthwise kernel: "
                + str(self.fixed_depthwise_kernel)
            )

        nn.init.zeros_(self.repair_u.weight)

    def forward(self, x):
        return (
            self.base(x)
            + self.repair_u(
                self.repair_d(
                    self.repair_v(x)
                )
            )
        )

    def delta_weight(self):
        v = self.repair_v.weight[:, :, 0, 0]
        d = self.repair_d.weight[:, 0, :, :]
        u = self.repair_u.weight[:, :, 0, 0]

        return torch.einsum(
            "or,ruv,ri->oiuv",
            u,
            d,
            v,
        )

    @torch.no_grad()
    def fused_conv(self):
        """Return analytically fused deployment Conv2d.

        The FLCR formula is unchanged:

            W_deploy = W + U D V

        Composition and accumulation are performed in FP64 and cast once
        back to the original Conv2d dtype. This reduces floating-point
        accumulation error without changing the deployed architecture,
        rank, placement, bias, stride, padding, or dilation.
        """
        fused = copy.deepcopy(self.base)

        with torch.no_grad():
            u64 = (
                self.repair_u.weight
                .detach()
                .double()[:, :, 0, 0]
            )

            d64 = (
                self.repair_d.weight
                .detach()
                .double()[:, 0, :, :]
            )

            v64 = (
                self.repair_v.weight
                .detach()
                .double()[:, :, 0, 0]
            )

            delta64 = torch.einsum(
                "or,ruv,ri->oiuv",
                u64,
                d64,
                v64,
            )

            deploy64 = (
                self.base.weight
                .detach()
                .double()
                + delta64
            )

            fused.weight.copy_(
                deploy64.to(
                    device=fused.weight.device,
                    dtype=fused.weight.dtype,
                )
            )

        return fused

    def adapter_parameters(self):
        yield from self.repair_v.parameters()
        yield from self.repair_d.parameters()
        yield from self.repair_u.parameters()


def _resolve_parent(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root

    for token in parts[:-1]:
        parent = getattr(parent, token)

    return parent, parts[-1]


def insert_flcr(
    model: nn.Module,
    layer_names: Iterable[str],
    rank: int = 8,
    fixed_depthwise_kernel=None,
):
    inserted: List[str] = []

    for name in layer_names:
        parent, leaf = _resolve_parent(model, name)
        old = getattr(parent, leaf)

        if isinstance(old, FLCRConv2d):
            raise RuntimeError(
                f"already FLCR: {name}"
            )

        if not isinstance(old, nn.Conv2d):
            raise TypeError(
                f"{name} is {type(old)}, expected Conv2d"
            )

        setattr(
            parent,
            leaf,
            FLCRConv2d(
                old,
                rank=rank,
                fixed_depthwise_kernel=fixed_depthwise_kernel,
            ),
        )

        inserted.append(name)

    return inserted


def fuse_flcr_inplace(model: nn.Module):
    replaced = []

    def visit(module: nn.Module, prefix=""):
        for key, child in list(module.named_children()):
            name = (
                f"{prefix}.{key}"
                if prefix else key
            )

            if isinstance(child, FLCRConv2d):
                setattr(
                    module,
                    key,
                    child.fused_conv(),
                )
                replaced.append(name)
            else:
                visit(child, name)

    visit(model)
    return replaced


def freeze_base_train_adapters(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []

    for name, module in model.named_modules():
        if not isinstance(module, FLCRConv2d):
            continue

        for local_name, p, is_trainable in [
            ("repair_v.weight", module.repair_v.weight, True),
            (
                "repair_d.weight",
                module.repair_d.weight,
                module.fixed_depthwise_kernel is None,
            ),
            ("repair_u.weight", module.repair_u.weight, True),
        ]:
            p.requires_grad_(is_trainable)
            if is_trainable:
                trainable.append(
                    f"{name}.{local_name}"
                )

    return trainable


def adapter_parameter_count(model: nn.Module):
    total = 0

    for module in model.modules():
        if isinstance(module, FLCRConv2d):
            total += sum(
                p.numel()
                for p in module.adapter_parameters()
            )

    return int(total)
