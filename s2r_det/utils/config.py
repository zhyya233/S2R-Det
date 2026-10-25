"""Resolve repair recipes with explicit, relative JSON inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from s2r_det.utils.io import load_json


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_recipe(path: Union[str, Path]) -> Dict[str, Any]:
    """Merge objects recursively; replace arrays/scalars, including explicit nulls.

    ``extends`` names one JSON file relative to the declaring recipe. Missing
    files, non-object recipes, invalid parent paths and inheritance cycles fail.
    The returned dictionary contains the effective settings, without ``extends``.
    """

    def resolve(current: Path, ancestors: Tuple[Path, ...]) -> Dict[str, Any]:
        current = current.resolve()
        if current in ancestors:
            raise ValueError("recipe inheritance cycle")
        data = load_json(current)
        if not isinstance(data, dict):
            raise ValueError("recipe must be a JSON object")
        parent = data.pop("extends", None)
        if parent is None:
            return data
        if not isinstance(parent, str) or not parent or Path(parent).is_absolute():
            raise ValueError("extends must name a nonempty relative JSON path")
        base = resolve(current.parent / parent, (*ancestors, current))
        return _merge(base, data)

    return resolve(Path(path), ())
