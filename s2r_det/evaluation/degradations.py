"""Deterministic S7 operators; interpolation/kernel choices are documented."""

import hashlib
import io
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

SETTINGS = {
    "gaussian_blur": [0.8, 1.6, 2.4],
    "gaussian_noise": [8.0, 16.0, 24.0],
    "jpeg": [60, 35, 15],
    "low_light": [0.75, 0.55, 0.35],
    "motion_blur": [5, 9, 13],
    "fog": [0.12, 0.25, 0.40],
}


def corrupt(rgb, condition, severity, image_identity, seed=0):
    if condition not in SETTINGS or severity not in (1, 2, 3):
        raise ValueError("unknown degradation/severity")
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    value = SETTINGS[condition][severity - 1]
    digest = hashlib.sha256(f"{seed}:{image_identity}:{condition}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    if condition == "gaussian_blur":
        return np.array(image.filter(ImageFilter.GaussianBlur(value)))
    if condition == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=value, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return np.array(compressed.convert("RGB"))
    if condition == "motion_blur":
        kernel = Image.new("L", (value, value), 0)
        draw = ImageDraw.Draw(kernel)
        center = value // 2
        angle = [0, 45, 90, 135][int.from_bytes(digest[8:10], "little") % 4]
        endpoints = {
            0: (0, center, value - 1, center),
            45: (0, value - 1, value - 1, 0),
            90: (center, 0, center, value - 1),
            135: (0, 0, value - 1, value - 1),
        }
        draw.line(endpoints[angle], fill=1, width=1)
        weights = np.array(kernel, dtype=float)
        weights /= weights.sum()
        # Pillow's Kernel supports sizes 3 and 5; motion kernels also need 9 and 13.
        arr = np.asarray(image, dtype=float)
        pad = value // 2
        extended = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(arr)
        for y, x in np.argwhere(weights):
            out += weights[y, x] * extended[y : y + arr.shape[0], x : x + arr.shape[1]]
    else:
        arr = np.asarray(image, dtype=float)
        out = (
            arr + rng.normal(0, value, arr.shape)
            if condition == "gaussian_noise"
            else arr * value if condition == "low_light" else arr * (1 - value) + 255 * value
        )
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def evaluate_profiles(dataset, image_loader, predict, seed=0):
    """Callbacks read decoded original RGB and return original-coordinate COCO rows."""
    from s2r_det.evaluation.coco import standard_metrics

    profiles = {}
    for condition in SETTINGS:
        values = []
        for severity in (1, 2, 3):
            predictions = []
            for image in dataset["images"]:
                degraded = corrupt(image_loader(image), condition, severity, str(image["id"]), seed)
                rows = predict(degraded, image)
                if any(r["image_id"] != image["id"] for r in rows):
                    raise ValueError("degradation predictor changed image identity")
                predictions.extend(rows)
            value = standard_metrics(dataset, predictions)["APsmall"]
            if value is None:
                raise ValueError("undefined degradation APsmall; no severity silently omitted")
            values.append(value)
        profiles[condition] = {"severity_APsmall": values, "mean_over_severities": sum(values) / 3}
    return {
        "profiles": profiles,
        "reference_model_seed": seed,
        "mean_unit": "severity levels, not repair-seed SD",
        "historical_pixel_equivalence": "UNVERIFIED",
    }
