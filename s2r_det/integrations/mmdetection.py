"""MMDetection3.3 RTMDet boundary; no dataset/checkpoint downloading."""

from __future__ import annotations
from pathlib import Path
import torch
from s2r_det.failure_atlas.atlas import trace_nms


class RTMDetAdapter:
    dense_convention = (
        "rtmdet-level-row-column; all-class logits; input-coordinate decoded xyxy; version1"
    )

    def __init__(self, config, checkpoint, device="cpu", category_ids=None):
        try:
            from mmengine.config import Config
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
        except ImportError as exc:
            raise ImportError(
                "RTMDet integration requires compatible MMDetection3.3/MMEngine/MMCV and PyTorch; see docs/usage.md"
            ) from exc
        if not Path(config).is_file() or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "local detector config and baseline checkpoint required; no automatic downloads"
            )
        cfg = Config.fromfile(config)

        def strip_init(node):
            if isinstance(node, dict):
                if "init_cfg" in node:
                    node["init_cfg"] = None
                if "pretrained" in node:
                    node["pretrained"] = None
                for value in node.values():
                    strip_init(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    strip_init(value)

        strip_init(cfg.model)
        register_all_modules(init_default_scope=True)
        self.model = init_detector(cfg, None, device=device)
        self.config = cfg
        from s2r_det.utils.io import fingerprint

        self.resolved_config_sha256 = fingerprint(cfg.to_dict())
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved.get("state_dict", saved)
        if not isinstance(state, dict) or not all(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError("provide a safe tensor state-dict checkpoint")
        if saved.get("format_version") == 1:
            meta = saved["metadata"]
            method = meta["recipe"]["method"]
            if not saved.get("fused", False) or method == "conv_adapter":
                from s2r_det.training.variants import configure_variant

                configure_variant(
                    self.model,
                    method,
                    meta.get("placement"),
                    meta["recipe"].get("rank", 8),
                    meta["recipe"].get("norm_paths"),
                )
        elif any("repair_" in n for n in state):
            raise ValueError(
                "repair checkpoints require format_version=1 and explicit recipe/placement metadata; baseline loading cannot accept repair keys"
            )
        self.model.load_state_dict(state, strict=True)
        if not hasattr(
            self.model, "bbox_head"
        ) or not self.model.bbox_head.__class__.__name__.startswith("RTMDet"):
            raise TypeError("this boundary supports RTMDet; YOLO requires its own adapter")
        self.head = self.model.bbox_head
        if getattr(self.model.data_preprocessor, "batch_augments", None):
            self.model.data_preprocessor.batch_augments = None
        self.category_ids = category_ids or list(range(1, self.head.num_classes + 1))
        if len(self.category_ids) != self.head.num_classes:
            raise ValueError("category mapping/model channel mismatch")

    def prepare(self, dataset, image_ids):
        records = []
        for image_id in image_ids:
            rgb, meta = dataset.load(image_id)
            records.append(
                {
                    "image_id": image_id,
                    "rgb": rgb,
                    "meta": meta,
                    "gt": dataset.gt(image_id),
                    "labels": dataset.labels,
                }
            )
        return self.prepare_records(records)

    def iter_prepared_batches(self, dataset, batches, seed, workers=8):
        from s2r_det.data.loading import raw_training_batches

        for records in raw_training_batches(dataset, batches, seed, workers):
            yield (
                {
                    "step": records[0]["step"],
                    "source": records[0]["source"],
                    "image_ids": [r["image_id"] for r in records],
                },
                self.prepare_records(records),
            )

    def prepare_records(self, records):
        import numpy as np
        from mmengine.structures import InstanceData
        from mmdet.structures import DetDataSample

        inputs = []
        samples = []
        for record in records:
            image_id = record["image_id"]
            image = record["rgb"]
            meta = record["meta"]
            # MMDetection's image-loader boundary convention is BGR.
            image = np.ascontiguousarray(image[:, :, ::-1])
            inputs.append(torch.from_numpy(image).permute(2, 0, 1))
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "img_id": image_id,
                    "ori_shape": (meta["original_height"], meta["original_width"]),
                    "img_shape": (meta["resized_height"], meta["resized_width"]),
                    "scale_factor": (meta["sx"], meta["sy"]),
                    "batch_input_shape": (640, 640),
                    "pad_shape": (640, 640),
                }
            )
            gt = record["gt"]
            boxes = torch.tensor([g["box"] for g in gt], dtype=torch.float32).reshape(-1, 4)
            boxes = boxes * torch.tensor([meta["sx"], meta["sy"], meta["sx"], meta["sy"]])
            labels = torch.tensor(
                [record["labels"][g["category_id"]] for g in gt], dtype=torch.long
            )
            sample.gt_instances = InstanceData(bboxes=boxes, labels=labels)
            samples.append(sample)
        return self.model.data_preprocessor(
            {"inputs": inputs, "data_samples": samples}, training=False
        )

    def dense(self, prepared):
        from mmdet.structures.bbox import distance2bbox

        raw = self.head(self.model.extract_feat(prepared["inputs"]))
        if len(raw) != 2:
            raise ValueError("unexpected RTMDet head outputs")
        scores, distances = raw
        batch = scores[0].shape[0]
        logits = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch, -1, self.head.cls_out_channels) for x in scores],
            1,
        )
        anchors, _ = self.head.get_anchors(
            [x.shape[-2:] for x in scores],
            [dict(s.metainfo) for s in prepared["data_samples"]],
            device=logits.device,
        )
        boxes = torch.stack(
            [
                torch.cat(
                    [
                        distance2bbox(a.reshape(-1, 4), p[b].permute(1, 2, 0).reshape(-1, 4))
                        for a, p in zip(anchors[b], distances)
                    ]
                )
                for b in range(batch)
            ]
        )
        return logits, boxes, raw

    def detection_loss(self, prepared):
        return self.model.loss(prepared["inputs"], prepared["data_samples"])

    @torch.no_grad()
    def candidates(self, prepared):
        """Native Ppre mapping plus an independently checked NMS/cap trace."""
        from mmdet.models.utils import filter_scores_and_topk

        if len(prepared["data_samples"]) != 1:
            raise ValueError("candidate trace is per image")
        _, boxes, raw = self.dense(prepared)
        scores, _ = raw
        cfg = self.model.test_cfg
        if cfg["nms"].get("type", "nms") != "nms" or cfg["nms"].get("class_agnostic", False):
            raise ValueError(
                "only configured hard class-wise NMS is supported; supply an explicit architecture adapter otherwise"
            )
        native_pre = self.head.predict_by_feat(
            *raw,
            batch_img_metas=[dict(prepared["data_samples"][0].metainfo)],
            cfg=cfg,
            rescale=False,
            with_nms=False,
        )[0]
        native_post = self.head.predict_by_feat(
            *raw,
            batch_img_metas=[dict(prepared["data_samples"][0].metainfo)],
            cfg=cfg,
            rescale=False,
            with_nms=True,
        )[0]
        entries = []
        offset = 0
        for level in scores:
            values = level[0].permute(1, 2, 0).reshape(-1, self.head.cls_out_channels).sigmoid()
            kept, labels, indices, _ = filter_scores_and_topk(
                values, float(cfg["score_thr"]), int(cfg.get("nms_pre", -1)), None
            )
            for score, cls, dense in zip(kept.tolist(), labels.tolist(), indices.tolist()):
                box = boxes[0, offset + dense].tolist()
                # Native decode clips to the transformed image dimensions.
                if getattr(self.head.bbox_coder, "clip_border", True):
                    height, width = prepared["data_samples"][0].metainfo["img_shape"]
                    box = [
                        max(0.0, min(float(width if i % 2 == 0 else height), v))
                        for i, v in enumerate(box)
                    ]
                minimum = float(cfg.get("min_bbox_size", -1))
                if minimum >= 0 and (box[2] - box[0] <= minimum or box[3] - box[1] <= minimum):
                    continue
                entries.append(
                    {
                        "score": score,
                        "category_id": self.category_ids[cls],
                        "box": box,
                        "dense_index": offset + dense,
                    }
                )
            offset += values.shape[0]
        if len(entries) != len(native_pre):
            raise RuntimeError(
                "native Ppre count mismatch; baseline inference policy needs reconciliation"
            )
        for p, b, s, c in zip(
            entries,
            native_pre.bboxes.tolist(),
            native_pre.scores.tolist(),
            native_pre.labels.tolist(),
        ):
            if (
                max(abs(x - y) for x, y in zip(p["box"], b)) > 1e-4
                or abs(p["score"] - s) > 1e-6
                or p["category_id"] != self.category_ids[c]
            ):
                raise RuntimeError("native Ppre mapping mismatch")
        trace = trace_nms(entries, float(cfg["nms"]["iou_threshold"]), int(cfg["max_per_img"]))
        actual = sorted(
            [
                (self.category_ids[c], s, *b)
                for c, s, b in zip(
                    native_post.labels.tolist(),
                    native_post.scores.tolist(),
                    native_post.bboxes.tolist(),
                )
            ]
        )
        computed = sorted(
            [
                (entries[i]["category_id"], entries[i]["score"], *entries[i]["box"])
                for i in trace["post_indices"]
            ]
        )
        if len(actual) != len(computed) or any(
            max(abs(x - y) for x, y in zip(a, b)) > 1e-4 for a, b in zip(actual, computed)
        ):
            raise RuntimeError("native NMS/cap trace mismatch; do not assign confirmed suppression")
        return entries, trace
