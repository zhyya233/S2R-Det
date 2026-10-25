"""Canonical architecture-aware records; insertion sets are declared inputs."""

import re
from collections import defaultdict
import torch
from s2r_det.adapters.flcr import module_references
from s2r_det.sensitivity.scoring import GROUPS


def build_manifest(model, input_tensor, insertion_sets, group_overrides=None):
    if not isinstance(insertion_sets, dict) or any(
        not isinstance(v, list) or any(not isinstance(p, str) for p in v)
        for v in insertion_sets.values()
    ):
        raise ValueError("insertion_sets must map semantic group names to lists of module paths")
    if set(insertion_sets) - set(GROUPS):
        raise ValueError("unknown semantic insertion group")
    refs = module_references(model)
    aliases = defaultdict(list)
    objects = {}
    for name, _, _, m in refs:
        if isinstance(m, torch.nn.Conv2d):
            aliases[id(m)].append(name)
            objects[id(m)] = m
    shapes = {}
    hooks = []
    training_states = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        for ident, m in objects.items():

            def hook(module, args, out, i=ident):
                shapes.setdefault(i, set()).add(tuple(out.shape[-2:]))

            hooks.append(m.register_forward_hook(hook))
        with torch.no_grad():
            model(input_tensor, mode="tensor")
    finally:
        for h in hooks:
            h.remove()
        for module, state in training_states:
            module.training = state
    rows = []
    group_overrides = group_overrides or {}
    for ident, names in aliases.items():
        name = names[0]
        m = objects[ident]
        grids = shapes.get(ident, set())
        if not grids:
            raise ValueError(f"eligible convolution not traced: {name}")
        strides = sorted(
            {(input_tensor.shape[-2] / h, input_tensor.shape[-1] / w) for h, w in grids}
        )
        group = group_overrides.get(name)
        if group is None:
            if name.startswith("backbone.stem"):
                group = "backbone_shallow"
            elif name.startswith("backbone.stage"):
                match = re.match(r"backbone.stage(\d+)\.", name)
                if match:
                    group = "backbone_shallow" if int(match[1]) <= 2 else "backbone_late"
            elif name.startswith("neck.") and len(strides) == 1:
                group = {(8.0, 8.0): "neck_p3", (16.0, 16.0): "neck_p4"}.get(strides[0])
            elif name.startswith("bbox_head.cls_convs.") or name.startswith("bbox_head.reg_convs."):
                group = "head"
        if group is not None and group not in GROUPS:
            raise ValueError("unknown semantic group override")
        eligible = group is not None and m.kernel_size == (3, 3) and m.groups == 1
        selected = any(n in insertion_sets.get(group, []) for n in names)
        if selected and not eligible:
            raise ValueError("declared insertion uses an unsupported or out-of-group operator")
        rows.append(
            {
                "path": name,
                "aliases": names,
                "shared_module_identity": name,
                "group": group,
                "in_channels": m.in_channels,
                "out_channels": m.out_channels,
                "kernel": list(m.kernel_size),
                "stride": list(m.stride),
                "padding": m.padding if isinstance(m.padding, str) else list(m.padding),
                "dilation": list(m.dilation),
                "groups": m.groups,
                "padding_mode": m.padding_mode,
                "output_strides": strides,
                "analysis_eligible": eligible,
                "ineligibility_reason": (
                    None
                    if eligible
                    else "outside semantic candidate groups or unsupported FLCR operator"
                ),
                "insertion_eligible": selected,
            }
        )
    declared = {n for names in insertion_sets.values() for n in names}
    available = {n for r in rows for n in r["aliases"]}
    if declared - available:
        raise ValueError(f"ineligible insertion paths: {sorted(declared-available)}")
    for r in rows:
        if any(
            n in names
            for g, names in insertion_sets.items()
            if g != r["group"]
            for n in r["aliases"]
        ):
            raise ValueError("insertion group conflicts with semantic membership")
    return rows
