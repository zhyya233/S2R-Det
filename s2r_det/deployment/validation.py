"""Finite-precision agreement and actual elapsed times are measured separately."""

import time
from s2r_det.failure_atlas.atlas import quantile


def compare_outputs(reference, candidate, tolerance=1e-4):
    import torch

    a = [reference] if isinstance(reference, torch.Tensor) else list(reference)
    b = [candidate] if isinstance(candidate, torch.Tensor) else list(candidate)
    if len(a) != len(b) or not a:
        raise ValueError("corresponding nonempty output lists required")
    maximum = 0.0
    for x, y in zip(a, b):
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("output shape/non-finite mismatch")
        if x.numel():
            maximum = max(maximum, float((x.double() - y.double()).abs().max()))
    return {
        "maximum_absolute_error": maximum,
        "tolerance": tolerance,
        "within_tolerance": maximum <= tolerance,
        "scope": "corresponding pre-NMS outputs only; not post-NMS/AP equivalence",
    }


def benchmark(
    predict_from_decoded_cpu_image,
    decoded_images,
    samples=1000,
    warmup=20,
    synchronize=lambda: None,
):
    if not decoded_images or samples < 1 or warmup < 0:
        raise ValueError("invalid latency profile")
    for i in range(warmup):
        predict_from_decoded_cpu_image(decoded_images[i % len(decoded_images)])
    times = []
    for i in range(samples):
        synchronize()
        start = time.perf_counter()
        predict_from_decoded_cpu_image(decoded_images[i % len(decoded_images)])
        synchronize()
        times.append((time.perf_counter() - start) * 1000)
    average = sum(times) / len(times)
    return {
        "samples": samples,
        "warmup": warmup,
        "P50_ms": quantile(times, 0.5),
        "P95_ms": quantile(times, 0.95),
        "mean_ms": average,
        "FPS": 1000 / average,
        "boundary": "callback must include preprocess/transfer/network/decode/NMS/final CPU detections; reading/decoding excluded",
        "warmup_provenance": "configurable reconstruction choice, not recovered historical setting",
    }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def profile_supported_flops(model, inputs):
    import torch

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as prof:
        with torch.no_grad():
            model(inputs)
    return {
        "GFLOPs_supported_ops": sum(e.flops for e in prof.key_averages()) / 1e9,
        "profiler": "torch.profiler with_flops; multiply/add convention of supported operators",
        "limitation": "untracked operators are not counted; this is not a complete detector FLOP total",
    }
