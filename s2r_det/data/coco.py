"""Retained COCO annotations and deterministic full-image geometry."""

import math
from collections import defaultdict
from pathlib import Path
from s2r_det.utils.io import load_json


def resize_geometry(width, height, size=640):
    if width <= 0 or height <= 0:
        raise ValueError("positive image dimensions required")
    scale = min(size / width, size / height)
    w = math.floor(width * scale + 0.5)
    h = math.floor(height * scale + 0.5)
    return w, h, w / width, h / height


class CocoImages:
    def __init__(self, annotations, image_root):
        self.dataset = load_json(annotations)
        self.root = Path(image_root)
        self.images = {str(i["id"]): i for i in self.dataset["images"]}
        if len(self.images) != len(self.dataset["images"]):
            raise ValueError("duplicate image IDs")
        self.categories = sorted(c["id"] for c in self.dataset["categories"])
        self.labels = {c: i for i, c in enumerate(self.categories)}
        self.annotations = defaultdict(list)
        keys = set()
        for a in self.dataset["annotations"]:
            k = (str(a["image_id"]), str(a["id"]))
            if k in keys or k[0] not in self.images:
                raise ValueError("duplicate GT key or unknown image")
            keys.add(k)
            if a.get("iscrowd", 0) or a.get("ignore", 0) or a["bbox"][2] <= 0 or a["bbox"][3] <= 0:
                raise ValueError("supply converted retained non-crowd annotations")
            self.annotations[k[0]].append(a)

    def gt(self, image_id):
        image = self.images[str(image_id)]
        _, _, sx, sy = resize_geometry(image["width"], image["height"])
        out = []
        for a in self.annotations[str(image_id)]:
            x, y, w, h = a["bbox"]
            out.append(
                {
                    "annotation_id": str(a["id"]),
                    "image_id": str(image_id),
                    "category_id": a["category_id"],
                    "box": [x, y, x + w, y + h],
                    "area": w * h,
                    "input_area": sx * sy * w * h,
                    "occlusion": a.get("occlusion"),
                }
            )
        return out

    def counts(self):
        return {i: len(self.annotations[i]) for i in self.images}

    def load(self, image_id):
        from PIL import Image
        import numpy as np

        im = self.images[str(image_id)]
        path = (self.root / im["file_name"]).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("image path escapes image root")
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            if rgb.size != (im["width"], im["height"]):
                raise ValueError("image/annotation geometry mismatch")
            w, h, sx, sy = resize_geometry(*rgb.size)
            canvas = Image.new("RGB", (640, 640), (114, 114, 114))
            canvas.paste(rgb.resize((w, h), Image.Resampling.BILINEAR), (0, 0))
        return np.array(canvas), {
            "image_id": str(image_id),
            "original_width": im["width"],
            "original_height": im["height"],
            "resized_width": w,
            "resized_height": h,
            "sx": sx,
            "sy": sy,
        }


def convert_visdrone(split_root):
    from PIL import Image

    root = Path(split_root)
    images = []
    annotations = []
    for image_path in sorted((root / "images").glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        image_id = len(images) + 1
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        for line_no, line in enumerate(
            (root / "annotations" / (image_path.stem + ".txt")).read_text().splitlines(), 1
        ):
            fields = line.rstrip(",").split(",")
            if len(fields) < 8:
                raise ValueError("expected eight VisDrone fields")
            x, y, w, h = map(float, fields[:4])
            score, category = int(fields[4]), int(fields[5])
            if score == 1 and 1 <= category <= 10 and w > 0 and h > 0:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "occlusion": int(fields[7]),
                        "source_row": line_no,
                    }
                )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": n}
            for i, n in enumerate(
                [
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
                ],
                1,
            )
        ],
    }
