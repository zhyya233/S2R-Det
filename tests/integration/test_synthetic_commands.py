"""Execute portable public workflows on tiny synthetic inputs, never paper data."""

import json
import pytest
from s2r_det.cli import main
from s2r_det.utils.io import file_sha256


def put(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def dataset():
    return {
        "images": [
            {
                "id": 1,
                "width": 640,
                "height": 640,
                "file_name": "sequence/frame.png",
                "sequence_id": "sequence",
            }
        ],
        "categories": [{"id": 1, "name": "toy"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }


def predictions():
    return [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]


def test_uavdt_preparation_and_evaluation(tmp_path):
    annotations = put(tmp_path / "gt.json", dataset())
    ready = str(tmp_path / "ready.json")
    main(["prepare-uavdt", "--annotations", annotations, "--output", ready])
    assert json.loads(open(ready).read())["images"][0]["sequence_id"] == "sequence"
    pred = put(tmp_path / "pred.json", predictions())
    out = str(tmp_path / "metrics.json")
    main(
        [
            "evaluate",
            "--annotations",
            annotations,
            "--predictions",
            pred,
            "--dataset",
            "uavdt",
            "--output",
            out,
        ]
    )
    assert json.loads(open(out).read())["equal_weight_sequence_AP70"] == pytest.approx(100)


def test_image_and_sequence_bootstrap_cli(tmp_path):
    annotations = put(tmp_path / "gt.json", dataset())
    b = put(tmp_path / "b.json", {"coco_predictions": predictions()})
    r = put(tmp_path / "r.json", {"coco_predictions": predictions()})
    out = str(tmp_path / "ci.json")
    main(
        [
            "bootstrap",
            "--unit",
            "image",
            "--annotations",
            annotations,
            "--baseline",
            b,
            "--repaired",
            r,
            "--replicates",
            "3",
            "--output",
            out,
        ]
    )
    assert json.loads(open(out).read())["CI95"] == [0, 0]
    b = put(tmp_path / "bseq.json", {"s1": 10, "s2": 20})
    r = put(tmp_path / "rseq.json", {"s1": 11, "s2": 21})
    main(
        [
            "bootstrap",
            "--unit",
            "sequence",
            "--baseline",
            b,
            "--repaired",
            r,
            "--replicates",
            "5",
            "--output",
            out,
        ]
    )
    assert json.loads(open(out).read())["CI95"] == [1, 1]


def test_sampling_support_and_seed_aggregation_cli(tmp_path):
    ids = put(tmp_path / "ids.json", ["a", "b", "c"])
    scores = put(tmp_path / "scores.json", {"a": 3, "b": 2, "c": 1})
    out = str(tmp_path / "support.json")
    main(
        [
            "sampling-support",
            "--image-ids",
            ids,
            "--count",
            "2",
            "--kind",
            "top_loss",
            "--scores",
            scores,
            "--output",
            out,
        ]
    )
    result = json.loads(open(out).read())
    assert result["image_ids"] == ["a", "b"] and result["score_source_identity"] == file_sha256(
        scores
    )
    records = put(tmp_path / "seeds.json", {"toy_metric": [1, 2, 3]})
    main(["aggregate-seeds", "--records", records, "--output", out])
    assert json.loads(open(out).read())["toy_metric"]["sample_SD"] == 1


def test_atlas_and_gt_diagnostics_cli(tmp_path):
    g = {
        "annotation_id": "g",
        "category_id": 1,
        "box": [0, 0, 10, 10],
        "area": 100,
        "input_area": 100,
    }
    image = {
        "image_id": "1",
        "gt": [g],
        "pre": [{"category_id": 1, "box": [0, 0, 10, 10], "score": 0.9}],
        "trace": {"post_indices": [0], "suppressed_indices": [], "capped_indices": []},
    }
    obj = {
        "images": [image],
        "annotations_sha256": "a",
        "checkpoint_sha256": "b",
        "config_sha256": "c",
    }
    path = put(tmp_path / "pred.json", obj)
    out = str(tmp_path / "atlas.json")
    main(
        [
            "atlas",
            "--predictions",
            path,
            "--split",
            "train",
            "--fit-training-quartiles",
            "--output",
            out,
        ]
    )
    assert json.loads(open(out).read())["category_counts"] == {"F3": 1}
    main(["diagnostics", "--baseline", path, "--repaired", path, "--output", out])
    result = json.loads(open(out).read())
    assert result["GT"] == 1 and result["net"] == 0 and result["success_retention_percent"] == 100


def test_conversion_keeps_only_retained_annotations(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    Image.new("RGB", (30, 20)).save(tmp_path / "images/toy.png")
    (tmp_path / "annotations/toy.txt").write_text(
        "0,0,10,10,1,1,0,2\n0,0,10,10,0,1,0,0\n0,0,10,10,1,11,0,0\n"
    )
    out = str(tmp_path / "retained.json")
    main(["convert-visdrone", "--split-root", str(tmp_path), "--output", out])
    result = json.loads(open(out).read())
    assert len(result["annotations"]) == 1 and result["annotations"][0]["occlusion"] == 2
