from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ConvAdapter(nn.Module):
    """Conv-Adapter bottleneck used for the P00 strong PEFT baseline.

    Frozen P00 choice:
      - Residual Parallel insertion
      - compression gamma=4
      - grouped spatial KxK down projection
      - GELU
      - pointwise up projection
      - learnable per-output-channel scaling alpha initialized to 1

    W_down follows the paper shape:
      [Cin/gamma, gamma, K, K]
    implemented by:
      Conv2d(Cin, Cin/gamma, groups=Cin/gamma)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        gamma: int = 4,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if gamma <= 0:
            raise ValueError("gamma must be positive")

        if in_channels % gamma != 0:
            raise ValueError(
                f"in_channels={in_channels} not divisible by gamma={gamma}"
            )

        mid = in_channels // gamma

        if mid <= 0:
            raise ValueError("invalid compressed channel count")

        padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gamma = gamma
        self.mid_channels = mid

        self.down = nn.Conv2d(
            in_channels,
            mid,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=mid,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.act = nn.GELU()

        self.up = nn.Conv2d(
            mid,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.alpha = nn.Parameter(
            torch.ones(
                out_channels,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x):
        y = self.up(
            self.act(
                self.down(x)
            )
        )

        return y * self.alpha.view(
            1, -1, 1, 1
        )


class ResidualParallelConvAdapter(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        gamma: int,
        device,
        dtype,
    ):
        super().__init__()

        self.base = base

        self.adapter = ConvAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            gamma=gamma,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        y = self.base(x)
        d = self.adapter(x)

        if d.shape != y.shape:
            raise RuntimeError(
                "Residual Parallel Conv-Adapter shape mismatch: "
                f"base={tuple(y.shape)}, adapter={tuple(d.shape)}"
            )

        return y + d


def _replace_module(root, name: str, module: nn.Module):
    parts = name.split(".")
    parent = root

    for part in parts[:-1]:
        parent = parent._modules[part]

    parent._modules[parts[-1]] = module


def insert_residual_parallel_conv_adapters(
    model: nn.Module,
    gamma: int = 4,
) -> List[str]:
    """Insert adapters into CSPNeXt residual blocks in the backbone.

    This realizes the paper's Residual Parallel dense-prediction baseline
    on the RTMDet CSPNeXt backbone.
    """

    candidates = []

    for name, module in model.backbone.named_modules():
        if module.__class__.__name__ != "CSPNeXtBlock":
            continue

        full_name = "backbone." + name if name else "backbone"

        convs = [
            m
            for m in module.modules()
            if isinstance(m, nn.Conv2d)
        ]

        if not convs:
            continue

        in_channels = convs[0].in_channels
        out_channels = convs[-1].out_channels

        spatial = [
            m
            for m in convs
            if (
                m.kernel_size[0] > 1
                or m.kernel_size[1] > 1
            )
        ]

        if not spatial:
            continue

        k = max(
            max(m.kernel_size)
            for m in spatial
        )

        if k % 2 == 0:
            raise RuntimeError(
                f"{full_name}: even spatial kernel {k} unsupported"
            )

        if in_channels % gamma != 0:
            raise RuntimeError(
                f"{full_name}: Cin={in_channels} not divisible "
                f"by gamma={gamma}"
            )

        p0 = next(module.parameters())

        candidates.append(
            (
                full_name,
                module,
                in_channels,
                out_channels,
                k,
                p0.device,
                p0.dtype,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No CSPNeXtBlock candidates found for Conv-Adapter"
        )

    names = []

    for (
        name,
        module,
        cin,
        cout,
        k,
        device,
        dtype,
    ) in candidates:

        wrapper = ResidualParallelConvAdapter(
            base=module,
            in_channels=cin,
            out_channels=cout,
            kernel_size=k,
            gamma=gamma,
            device=device,
            dtype=dtype,
        )

        _replace_module(
            model,
            name,
            wrapper,
        )

        names.append(name)

    return names


def freeze_base_train_conv_adapters(
    model: nn.Module,
) -> List[str]:

    for p in model.parameters():
        p.requires_grad = False

    trainable = []

    for name, p in model.named_parameters():
        if ".adapter." in name:
            p.requires_grad = True
            trainable.append(name)

    if not trainable:
        raise RuntimeError(
            "No Conv-Adapter parameters were made trainable"
        )

    return trainable
