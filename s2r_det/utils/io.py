"""Strict portable JSON and artifact identities."""

import hashlib
import json
from pathlib import Path


def load_json(path):
    def reject(v):
        raise ValueError(f"non-finite JSON constant: {v}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require_identity(actual, expected):
    keys = {"baseline", "annotations", "preprocessing", "dense_convention", "atlas"}
    if set(actual) != keys or set(expected) != keys:
        raise ValueError(
            "cache must bind baseline, annotations, preprocessing, dense_convention and atlas"
        )
    if any(
        not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in actual.values()
    ):
        raise ValueError("identity values must be SHA-256 digests")
    if actual != expected:
        raise ValueError("teacher-cache identity mismatch")
