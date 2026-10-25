"""Optional execution boundaries fail explicitly when unavailable."""


def onnx_session(path, providers=None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX Runtime is optional and is not installed") from exc
    return ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])


def tensor_rt_engine(factory=None, **kwargs):
    import importlib

    try:
        importlib.import_module("tensorrt")
    except ImportError as exc:
        raise ImportError("TensorRT/CUDA optional backend is unavailable") from exc
    if factory is None:
        raise ValueError(
            "provide an explicit validated TensorRT engine/buffer adapter; no engine or weights are distributed"
        )
    return factory(**kwargs)
