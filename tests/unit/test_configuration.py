"""Recipe inheritance preserves explicit controls and rejects invalid inputs."""

import json
from pathlib import Path

import pytest

from s2r_det.training.protocol import RepairProtocol
from s2r_det.utils.config import load_recipe


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def test_relative_inheritance_and_explicit_replacements(tmp_path):
    write_json(
        tmp_path / "base.json",
        {"protocol": {"steps": 24300, "learning_rate": 0.0003}, "allowed_seeds": [0, 1, 2]},
    )
    child = tmp_path / "nested"
    child.mkdir()
    recipe = write_json(
        child / "repair.json",
        {"extends": "../base.json", "protocol": {"learning_rate": None}, "allowed_seeds": [0]},
    )
    assert load_recipe(recipe) == {
        "protocol": {"steps": 24300, "learning_rate": None},
        "allowed_seeds": [0],
    }
    assert load_recipe(tmp_path / "base.json")["protocol"]["learning_rate"] == 0.0003


def test_inheritance_cycle(tmp_path):
    first = write_json(tmp_path / "first.json", {"extends": "second.json"})
    write_json(tmp_path / "second.json", {"extends": "first.json"})
    with pytest.raises(ValueError, match="cycle"):
        load_recipe(first)


@pytest.mark.parametrize("parent", ["", 1, ["base.json"], "/base.json"])
def test_invalid_parent_path(tmp_path, parent):
    recipe = write_json(tmp_path / "child.json", {"extends": parent})
    with pytest.raises(ValueError, match="relative JSON path"):
        load_recipe(recipe)


def test_missing_parent_and_non_object(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recipe(write_json(tmp_path / "missing.json", {"extends": "absent.json"}))
    with pytest.raises(ValueError, match="JSON object"):
        load_recipe(write_json(tmp_path / "array.json", []))


def test_indexed_recipes_resolve_and_keep_distinct_budgets():
    root = Path(__file__).resolve().parents[2]
    index = json.loads((root / "configs/experiments/index.json").read_text())
    recipes = {
        name: load_recipe(root / "configs/experiments" / (name + ".json"))
        for name in index["recipes"]
    }
    for recipe in recipes.values():
        values = dict(recipe["protocol"])
        if values["learning_rate"] is None:
            continue
        RepairProtocol(**values)
    assert recipes["full_finetuning_equal_time"]["protocol"]["steps"] == 25245
    assert recipes["full_finetuning_equal_steps"]["protocol"]["steps"] == 42525
    assert recipes["factorial_uniform_retention_0"]["allowed_seeds"] == [0, 1, 2]
    assert recipes["sampling_uniform"]["allowed_seeds"] == [0]
    assert (
        recipes["factorial_failure_stratified_retention_0_25"]["sampling"]["failure_fraction"]
        == 0.7
    )
