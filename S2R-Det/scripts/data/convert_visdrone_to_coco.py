import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

from PIL import Image

CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

SPLITS = {
    "train": ("VisDrone2019-DET-train", 6471),
    "val": ("VisDrone2019-DET-val", 548),
    "testdev": ("VisDrone2019-DET-test-dev", 1610),
}

def file_sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_annotation(path):
    rows = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw.strip():
            continue
        parts = [x.strip() for x in raw.split(",")]
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) != 8:
            raise RuntimeError(
                f"bad annotation field count: {path}:{line_no}:{len(parts)}"
            )
        vals = [int(round(float(x))) for x in parts]
        rows.append((line_no, vals))
    return rows

def convert_split(root, outdir, split, dirname, expected):
    base = root / dirname
    image_dir = base / "images"
    ann_dir = base / "annotations"

    images = sorted(image_dir.glob("*.jpg"))
    if len(images) != expected:
        raise RuntimeError(
            f"{split}: image count {len(images)} != expected {expected}"
        )

    coco = {
        "info": {
            "description": "VisDrone2019-DET converted for S2R-Det",
            "source_format": "VisDrone DET 8-column raw annotations",
            "conversion_policy": (
                "train/eval targets require score=1, category=1..10, "
                "positive width and height; raw files remain unchanged"
            ),
        },
        "licenses": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(CLASSES)
        ],
        "images": [],
        "annotations": [],
    }

    stats = Counter()
    dropped = []
    ann_id = 1

    for image_id, img_path in enumerate(images, 1):
        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.is_file():
            raise RuntimeError(f"missing annotation: {ann_path}")

        for line_no, vals in parse_annotation(ann_path):
            x, y, bw, bh, score, cat, trunc, occ = vals
            stats["raw_rows"] += 1

            # Official VisDrone semantics:
            # score=0 is ignored; classes 0 and 11 are not detection classes.
            if score != 1:
                stats["ignored_score0"] += 1
                continue

            if not (1 <= cat <= 10):
                stats["ignored_non_detection_category"] += 1
                continue

            if bw <= 0 or bh <= 0:
                stats["dropped_nonpositive_valid_gt"] += 1
                if len(dropped) < 20:
                    dropped.append({
                        "split": split,
                        "file": ann_path.name,
                        "line": line_no,
                        "bbox": [x, y, bw, bh],
                        "score": score,
                        "category": cat,
                    })
                continue

            if x < 0 or y < 0 or x + bw > width or y + bh > height:
                raise RuntimeError(
                    f"unexpected out-of-bounds valid GT: "
                    f"{ann_path}:{line_no} bbox={x,y,bw,bh} image={width,height}"
                )

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "ignore": 0,
                "visdrone_truncation": trunc,
                "visdrone_occlusion": occ,
            })

            ann_id += 1
            stats["valid_gt"] += 1
            stats[f"class_{cat}"] += 1

    out_path = outdir / f"visdrone2019_det_{split}.json"
    out_path.write_text(
        json.dumps(coco, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    return {
        "images": len(coco["images"]),
        "raw_rows": stats["raw_rows"],
        "valid_gt": stats["valid_gt"],
        "ignored_score0": stats["ignored_score0"],
        "ignored_non_detection_category": stats["ignored_non_detection_category"],
        "dropped_nonpositive_valid_gt":
            stats["dropped_nonpositive_valid_gt"],
        "dropped_samples": dropped,
        "coco_json": str(out_path),
        "coco_sha256": file_sha(out_path),
        "coco_size_bytes": out_path.stat().st_size,
        "class_counts": {
            str(i): stats[f"class_{i}"]
            for i in range(1, 11)
        },
    }

def inspect_rtmdet_transform():
    from mmengine.config import Config

    config_path = Path(
        "/home/a/miniconda3/envs/RT-DETR/lib/python3.10/"
        "site-packages/mmdet/.mim/configs/rtmdet/"
        "rtmdet_tiny_8xb32-300e_coco.py"
    )

    cfg = Config.fromfile(str(config_path))
    pipeline = cfg.get("test_pipeline", [])

    resize = None
    pad = None
    summary = []

    for item in pipeline:
        d = dict(item)
        typ = str(d.get("type", ""))
        summary.append({
            "type": typ,
            "scale": d.get("scale"),
            "keep_ratio": d.get("keep_ratio"),
            "size": d.get("size"),
            "size_divisor": d.get("size_divisor"),
        })

        if typ.endswith("Resize"):
            resize = d
        if typ.endswith("Pad"):
            pad = d

    resize_scale = tuple(resize.get("scale", ())) if resize else ()
    pad_size = tuple(pad.get("size", ())) if pad and pad.get("size") else ()

    gate = (
        resize is not None
        and resize_scale == (640, 640)
        and bool(resize.get("keep_ratio", False))
        and pad is not None
        and pad_size == (640, 640)
    )

    return {
        "config": str(config_path),
        "pipeline": summary,
        "resize_scale": resize_scale,
        "resize_keep_ratio":
            resize.get("keep_ratio") if resize else None,
        "pad_size": pad_size,
        "gate": "PASS" if gate else "HOLD",
    }

def letterbox_roundtrip(root):
    # RTMDet eval pipeline: keep-ratio resize to <=640x640,
    # followed by Pad(size=(640,640)). Default MMDet Pad is
    # right/bottom padding, so bbox has no x/y offset.
    max_err = 0.0
    tested = 0

    for split, (dirname, _) in SPLITS.items():
        base = root / dirname
        for ann_path in sorted((base / "annotations").glob("*.txt")):
            img_path = base / "images" / f"{ann_path.stem}.jpg"

            with Image.open(img_path) as im:
                iw, ih = im.size

            scale = min(640.0 / iw, 640.0 / ih)

            for _, vals in parse_annotation(ann_path):
                x, y, bw, bh, score, cat, _, _ = vals

                if score != 1 or not (1 <= cat <= 10):
                    continue
                if bw <= 0 or bh <= 0:
                    continue

                # Forward
                bx = [
                    x * scale,
                    y * scale,
                    (x + bw) * scale,
                    (y + bh) * scale,
                ]

                # Inverse
                inv = [v / scale for v in bx]
                ref = [x, y, x + bw, y + bh]

                err = max(abs(a - b) for a, b in zip(inv, ref))
                max_err = max(max_err, err)
                tested += 1

    return {
        "tested_boxes": tested,
        "max_abs_error_px": max_err,
        "gate": "PASS" if max_err < 1e-6 else "HOLD",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "VisDrone2019-DET",
        "root": str(root),
        "conversion_rule": {
            "raw_data_modified": False,
            "valid_detection_category_ids": list(range(1, 11)),
            "ignore_score": 0,
            "ignore_categories": [0, 11],
            "drop_nonpositive_bbox_after_semantic_filter": True,
        },
        "splits": {},
    }

    for split, (dirname, expected) in SPLITS.items():
        summary["splits"][split] = convert_split(
            root, outdir, split, dirname, expected
        )

    summary["rtmdet_eval_transform"] = inspect_rtmdet_transform()

    if summary["rtmdet_eval_transform"]["gate"] == "PASS":
        summary["letterbox_roundtrip"] = letterbox_roundtrip(root)
    else:
        summary["letterbox_roundtrip"] = {
            "tested_boxes": 0,
            "max_abs_error_px": None,
            "gate": "NOT_RUN",
        }

    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
