"""Native YOLO repair contract for user-supplied integrations."""

from typing import Protocol


class YOLORepairAdapter(Protocol):
    """External integration must implement all operations, not just inference.

    Required: torch model, prepare(full GT), native detection_loss, differentiable
    dense logits+input boxes, diagnostic_loss with declared native objectives,
    category mapping and class-wise pre/post-NMS provenance. YOLO objectness,
    decoder and normalization semantics must be explicitly documented.
    """

    model: object
    dense_convention: str
    resolved_config_sha256: str

    def prepare(self, dataset, image_ids): ...
    def dense(self, prepared): ...
    def detection_loss(self, prepared): ...
    def candidates(self, prepared): ...
    def diagnostic_loss(self, logits, boxes, gt_box, class_index): ...


def load_external_adapter(factory, **kwargs):
    import importlib

    if ":" not in factory:
        raise ValueError("external factory must be module:function")
    module, name = factory.split(":", 1)
    obj = getattr(importlib.import_module(module), name)(**kwargs)
    for attribute in [
        "model",
        "dense_convention",
        "prepare",
        "dense",
        "detection_loss",
        "candidates",
        "diagnostic_loss",
        "resolved_config_sha256",
    ]:
        if not hasattr(obj, attribute):
            raise TypeError(f"architecture integration missing {attribute}")
    return obj
