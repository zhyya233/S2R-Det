"""Linear parallel repair; preserve base/repair_v/repair_d/repair_u state keys."""

from __future__ import annotations
import copy
import torch
from torch import nn


class FLCRConv2d(nn.Module):
    def __init__(self, base, rank=8, fixed_depthwise_kernel=None):
        super().__init__()
        if not isinstance(base, nn.Conv2d) or base.kernel_size != (3, 3) or base.groups != 1:
            raise ValueError("FLCR requires an ordinary groups=1 3x3 Conv2d")
        if isinstance(rank, bool) or int(rank) != rank or rank < 1:
            raise ValueError("rank must be a positive integer")
        if base.padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("unsupported boundary handling")
        self.base = base
        self.rank = int(rank)
        self.fixed_depthwise_kernel = fixed_depthwise_kernel
        base.requires_grad_(False)
        kw = dict(device=base.weight.device, dtype=base.weight.dtype, bias=False)
        self.repair_v = nn.Conv2d(base.in_channels, self.rank, 1, **kw)
        self.repair_d = nn.Conv2d(
            self.rank,
            self.rank,
            3,
            groups=self.rank,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            padding_mode=base.padding_mode,
            **kw,
        )
        self.repair_u = nn.Conv2d(self.rank, base.out_channels, 1, **kw)
        if fixed_depthwise_kernel is not None:
            if fixed_depthwise_kernel != "laplacian4":
                raise ValueError("unsupported fixed spatial kernel")
            with torch.no_grad():
                self.repair_d.weight.copy_(
                    base.weight.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).expand(
                        self.rank, 1, 3, 3
                    )
                )
            self.repair_d.requires_grad_(False)
        nn.init.zeros_(self.repair_u.weight)

    def forward(self, x):
        return self.base(x) + self.repair_u(self.repair_d(self.repair_v(x)))

    def delta_weight(self):
        return torch.einsum(
            "or,ruv,ri->oiuv",
            self.repair_u.weight[:, :, 0, 0],
            self.repair_d.weight[:, 0],
            self.repair_v.weight[:, :, 0, 0],
        )

    @torch.no_grad()
    def fused_conv(self):
        fused = copy.deepcopy(self.base)
        delta = torch.einsum(
            "or,ruv,ri->oiuv",
            self.repair_u.weight.double()[:, :, 0, 0],
            self.repair_d.weight.double()[:, 0],
            self.repair_v.weight.double()[:, :, 0, 0],
        )
        fused.weight.copy_((self.base.weight.double() + delta).to(fused.weight.dtype))
        return fused

    def adapter_parameters(self):
        for branch in (self.repair_v, self.repair_d, self.repair_u):
            yield from branch.parameters()


def module_references(model):
    refs = []

    def visit(parent, prefix, ancestors):
        for key, child in parent._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{key}" if prefix else key
            refs.append((path, parent, key, child))
            if id(child) in ancestors:
                raise ValueError("cyclic module graph unsupported")
            visit(child, path, ancestors | {id(child)})

    visit(model, "", {id(model)})
    return refs


def insert_flcr(model, layer_names, rank=8, fixed_depthwise_kernel=None):
    refs = module_references(model)
    by_name = {n: m for n, _, _, m in refs}
    names = list(layer_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("provide unique nonempty insertion paths")
    if set(names) - by_name.keys():
        raise ValueError("unknown insertion path")
    targets = {id(by_name[n]): by_name[n] for n in names}
    wrappers = {i: FLCRConv2d(m, rank, fixed_depthwise_kernel) for i, m in targets.items()}
    for _, parent, key, m in refs:
        if id(m) in wrappers:
            parent._modules[key] = wrappers[id(m)]
    return names


def fuse_flcr_inplace(model):
    refs = module_references(model)
    fused = {id(m): m.fused_conv() for _, _, _, m in refs if isinstance(m, FLCRConv2d)}
    replaced = []
    for name, parent, key, m in refs:
        if id(m) in fused:
            parent._modules[key] = fused[id(m)]
            replaced.append(name)
    if any(isinstance(m, FLCRConv2d) for m in model.modules()):
        raise RuntimeError("fusion left repair branches")
    return replaced


def freeze_base_train_adapters(model):
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, FLCRConv2d):
            m.repair_v.requires_grad_(True)
            m.repair_u.requires_grad_(True)
            m.repair_d.requires_grad_(m.fixed_depthwise_kernel is None)
    return [n for n, p in model.named_parameters() if p.requires_grad]


def adapter_parameter_count(model):
    params = {
        id(p): p
        for m in model.modules()
        if isinstance(m, FLCRConv2d)
        for p in m.adapter_parameters()
        if p.requires_grad
    }
    return sum(p.numel() for p in params.values())
