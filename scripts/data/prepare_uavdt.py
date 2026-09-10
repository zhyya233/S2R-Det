from pathlib import Path
from collections import Counter
from PIL import Image
from mmengine.config import Config
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys

ROOT = Path("/home/a/projects/S2R-Det")
BASE = Path("/home/a/projects/datasets/UAVDT")
RAW = BASE / "raw"
IMGROOT = RAW / "UAV-benchmark-M"
TOOLKIT = RAW / "UAV-benchmark-MOTD_v1.0"
GTROOT = TOOLKIT / "GT"
ATTRROOT = RAW / "M_attr"
DERIVED = BASE / "derived/coco"
DERIVED.mkdir(parents=True, exist_ok=True)

ARCH = BASE / "_archives"

EXPECTED_ARCHIVE_SHA = {
    "UAV-benchmark-M.zip":
        "e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08",
    "UAV-benchmark-MOTD_v1.0.zip":
        "1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799",
    "M_attr.zip":
        "ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1",
}

OFFICIAL_TEST = [
    "M0203","M0205","M0208","M0209","M0403",
    "M0601","M0602","M0606","M0701","M0801",
    "M0802","M1001","M1004","M1007","M1009",
    "M1101","M1301","M1302","M1303","M1401",
]

CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "bus",
}

ATTRIBUTE_NAMES = [
    "daylight", "night", "fog",
    "low_alt", "medium_alt", "high_alt",
    "front_view", "side_view", "bird_view",
    "long_term",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for name, expected in EXPECTED_ARCHIVE_SHA.items():
    actual = sha256(ARCH / name)
    if actual != expected:
        raise RuntimeError(
            f"archive hash mismatch {name}: {actual}"
        )

seqs = sorted(
    p.name for p in IMGROOT.iterdir()
    if p.is_dir() and re.fullmatch(r"M\d{4}", p.name)
)
if len(seqs) != 50:
    raise RuntimeError(f"sequence count {len(seqs)} != 50")

# Official split is independently supported by M_attr train/test and
# CalculateDetectionPR_overall.m. Normalize only the known filename
# whitespace anomaly; raw files are never renamed.
def attr_sequence(path):
    x = path.name[:-len("_attr.txt")].strip()
    if not re.fullmatch(r"M\d{4}", x):
        raise RuntimeError(f"bad attribute filename: {path}")
    return x

train_seqs = sorted(
    attr_sequence(p)
    for p in (ATTRROOT / "train").glob("*_attr.txt")
)
test_seqs = sorted(
    attr_sequence(p)
    for p in (ATTRROOT / "test").glob("*_attr.txt")
)

if len(train_seqs) != 30 or len(test_seqs) != 20:
    raise RuntimeError("official attribute split is not 30/20")

if set(train_seqs) & set(test_seqs):
    raise RuntimeError("train/test sequence overlap")

if set(train_seqs) | set(test_seqs) != set(seqs):
    raise RuntimeError("split does not cover all sequences")

if test_seqs != sorted(OFFICIAL_TEST):
    raise RuntimeError("M_attr test split != official evaluator test split")

# Sequence attributes.
attributes = {}
raw_attribute_files = {}

for split in ("train", "test"):
    for p in sorted((ATTRROOT / split).glob("*_attr.txt")):
        s = attr_sequence(p)
        vals = [
            int(x.strip())
            for x in p.read_text(
                encoding="utf-8-sig",
                errors="replace",
            ).strip().split(",")
        ]
        if len(vals) != 10 or any(x not in (0, 1) for x in vals):
            raise RuntimeError(f"invalid attribute vector {p}")
        attributes[s] = dict(zip(ATTRIBUTE_NAMES, vals))
        raw_attribute_files[s] = str(p.relative_to(BASE))

# Frame count and dimensions. The prior raw gate already scanned all
# image headers; here we verify deterministic sequence-level dimensions.
frame_counts = {}
sequence_sizes = {}

for s in seqs:
    imgs = sorted((IMGROOT / s).glob("img*.jpg"))
    nums = []
    for p in imgs:
        m = re.fullmatch(r"img(\d+)\.jpg", p.name)
        if not m:
            raise RuntimeError(f"bad image filename {p}")
        nums.append(int(m.group(1)))

    if nums != list(range(1, len(nums) + 1)):
        raise RuntimeError(f"non-contiguous frames in {s}")

    frame_counts[s] = len(imgs)

    with Image.open(imgs[0]) as im:
        sequence_sizes[s] = tuple(im.size)

if sum(frame_counts.values()) != 40735:
    raise RuntimeError("frame total != 40735")

if sum(frame_counts[s] for s in test_seqs) != 16592:
    raise RuntimeError("official test frame total != 16592")

def read_gt(path):
    rows = []
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            vals = [x.strip() for x in line.split(",")]
            if len(vals) != 9:
                raise RuntimeError(
                    f"{path}:{lineno} does not have 9 columns"
                )
            nums = [float(x) for x in vals]
            rows.append(nums)
    return rows

categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "truck"},
    {"id": 3, "name": "bus"},
]

split_outputs = {}
overall_stats = {}

for split, split_seqs in (
    ("train", train_seqs),
    ("test", test_seqs),
):
    images = []
    anns = []
    ann_id = 1
    image_id = 1

    cat_counts = Counter()
    out_counts = Counter()
    occ_counts = Counter()
    out_of_bounds = 0

    for s in split_seqs:
        width, height = sequence_sizes[s]
        whole = read_gt(GTROOT / f"{s}_gt_whole.txt")

        frame_to_image_id = {}

        for frame in range(1, frame_counts[s] + 1):
            fname = f"{s}/img{frame:06d}.jpg"
            frame_to_image_id[frame] = image_id

            images.append({
                "id": image_id,
                "file_name": fname,
                "width": width,
                "height": height,
                "sequence_id": s,
                "frame_index": frame,
                "attributes": attributes[s],
            })
            image_id += 1

        for r in whole:
            frame = int(r[0])
            track_id = int(r[1])
            x, y, w, h = [float(v) for v in r[2:6]]
            out_view = int(r[6])
            occlusion = int(r[7])
            category_id = int(r[8])

            if category_id not in CATEGORY_NAMES:
                raise RuntimeError(
                    f"invalid category {category_id} in {s}"
                )
            if frame not in frame_to_image_id:
                raise RuntimeError(
                    f"invalid frame {frame} in {s}"
                )
            if w <= 0 or h <= 0:
                raise RuntimeError(
                    f"non-positive bbox in {s} frame {frame}"
                )

            # Preserve official raw box geometry. Do not clip:
            # out-of-view is an official diagnostic attribute.
            if (
                x < 0 or y < 0
                or x + w > width
                or y + h > height
            ):
                out_of_bounds += 1

            anns.append({
                "id": ann_id,
                "image_id": frame_to_image_id[frame],
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "track_id": track_id,
                "out_of_view": out_view,
                "occlusion": occlusion,
                "sequence_id": s,
                "frame_index": frame,
            })
            ann_id += 1

            cat_counts[category_id] += 1
            out_counts[out_view] += 1
            occ_counts[occlusion] += 1

    data = {
        "info": {
            "dataset": "UAVDT",
            "split": split,
            "source": "official UAV-benchmark-M + MOTD_v1.0",
            "conversion_policy":
                "preserve *_gt_whole.txt geometry; no clipping; "
                "official ignore regions remain external and are "
                "handled by the official UAVDT evaluator",
        },
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

    out = DERIVED / f"{split}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    tmp.replace(out)

    split_outputs[split] = {
        "path": str(out),
        "sha256": sha256(out),
        "images": len(images),
        "annotations": len(anns),
    }

    overall_stats[split] = {
        "sequences": len(split_seqs),
        "frames": len(images),
        "annotations": len(anns),
        "category_counts": dict(sorted(cat_counts.items())),
        "out_of_view_counts": dict(sorted(out_counts.items())),
        "occlusion_counts": dict(sorted(occ_counts.items())),
        "raw_box_out_of_bounds_count": out_of_bounds,
    }

# Preserve ignore regions as a separate derived artifact for diagnostics.
ignore = {}
for s in seqs:
    rows = read_gt(GTROOT / f"{s}_gt_ignore.txt")
    ignore[s] = [
        {
            "frame_index": int(r[0]),
            "track_id": int(r[1]),
            "bbox": [float(v) for v in r[2:6]],
        }
        for r in rows
    ]

ignore_path = DERIVED / "ignore_regions.json"
ignore_path.write_text(
    json.dumps(ignore, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)

archive_sha = {
    name: sha256(ARCH / name)
    for name in EXPECTED_ARCHIVE_SHA
}

raw_manifest = {
    "dataset": "UAVDT",
    "status": "RAW_INTAKE_PASS",
    "raw_root": str(RAW),
    "archive_sha256": archive_sha,
    "sequence_count": 50,
    "frame_count": 40735,
    "gt_whole_rows": 798795,
    "categories": CATEGORY_NAMES,
    "image_sizes": {
        "960x540": 1906,
        "1024x540": 38829,
    },
    "attribute_semantics": {
        "order": ATTRIBUTE_NAMES,
        "weather": ["daylight", "night", "fog"],
        "altitude": ["low_alt", "medium_alt", "high_alt"],
        "viewpoint": [
            "front_view", "side_view", "bird_view"
        ],
        "viewpoint_is_multilabel": True,
        "long_term_is_independent": True,
        "known_raw_filename_anomaly":
            "M_attr/test/M0701 _attr.txt",
        "raw_files_renamed": False,
    },
    "official_evaluator": {
        "overall":
            "raw/UAV-benchmark-MOTD_v1.0/"
            "utils/CalculateDetectionPR_overall.m",
        "sequence_attributes":
            "raw/UAV-benchmark-MOTD_v1.0/"
            "utils/CalculateDetectionPR_seq.m",
        "object_attributes":
            "raw/UAV-benchmark-MOTD_v1.0/"
            "utils/CalculateDetectionPR_obj.m",
        "primary_metric":
            "official UAVDT VOC-style AP at IoU=0.5",
    },
}

split_manifest = {
    "dataset": "UAVDT",
    "status": "OFFICIAL_SEQUENCE_SPLIT_FROZEN",
    "source":
        "M_attr train/test directories cross-checked against "
        "CalculateDetectionPR_overall.m",
    "train_sequences": train_seqs,
    "test_sequences": test_seqs,
    "train_sequence_count": 30,
    "test_sequence_count": 20,
    "train_frames": sum(frame_counts[s] for s in train_seqs),
    "test_frames": sum(frame_counts[s] for s in test_seqs),
    "sequence_disjoint": True,
    "neighbor_frame_leakage_possible": False,
    "frame_counts": frame_counts,
    "raw_attribute_files": raw_attribute_files,
}

conversion_manifest = {
    "dataset": "UAVDT",
    "status": "COCO_CONVERSION_PASS",
    "conversion_source": "*_gt_whole.txt",
    "category_mapping": CATEGORY_NAMES,
    "bbox_policy": "preserve raw xywh; no clipping",
    "ignore_policy":
        "not injected into COCO categories; preserve separately and "
        "use official UAVDT ignore semantics for primary evaluation",
    "test_set_use":
        "evaluation only; never used for checkpoint selection or tuning",
    "outputs": split_outputs,
    "ignore_regions": {
        "path": str(ignore_path),
        "sha256": sha256(ignore_path),
    },
    "statistics": overall_stats,
}

manifest_dir = ROOT / "data/manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)

(manifest_dir / "uavdt_raw_manifest.json").write_text(
    json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
(manifest_dir / "uavdt_official_split.json").write_text(
    json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
(manifest_dir / "uavdt_coco_conversion.json").write_text(
    json.dumps(conversion_manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# Build UAVDT baseline config from the already frozen VisDrone baseline
# configuration so architecture and training method remain aligned.
base_cfg_path = ROOT / "configs/visdrone/rtmdet_tiny_baseline.py"
cfg = Config.fromfile(str(base_cfg_path))

metainfo = {
    "classes": ("car", "truck", "bus"),
}

cfg.model["bbox_head"]["num_classes"] = 3

data_root = "data/links/uavdt/"

def patch_dataset(ds, ann_file, test_mode):
    if not isinstance(ds, dict):
        return

    if "dataset" in ds and isinstance(ds["dataset"], dict):
        patch_dataset(ds["dataset"], ann_file, test_mode)
        return

    ds["type"] = "CocoDataset"
    ds["data_root"] = data_root
    ds["ann_file"] = ann_file
    ds["data_prefix"] = {
        "img": "raw/UAV-benchmark-M/"
    }
    ds["metainfo"] = metainfo
    ds["test_mode"] = test_mode

patch_dataset(
    cfg.train_dataloader["dataset"],
    "derived/coco/train.json",
    False,
)
patch_dataset(
    cfg.val_dataloader["dataset"],
    "derived/coco/test.json",
    True,
)
patch_dataset(
    cfg.test_dataloader["dataset"],
    "derived/coco/test.json",
    True,
)

def patch_eval(ev):
    if isinstance(ev, dict):
        if ev.get("type") == "CocoMetric":
            ev["ann_file"] = (
                data_root + "derived/coco/test.json"
            )
        for v in ev.values():
            patch_eval(v)
    elif isinstance(ev, list):
        for v in ev:
            patch_eval(v)

patch_eval(cfg.val_evaluator)
patch_eval(cfg.test_evaluator)

# UAVDT official test is never validation for model selection.
# Train the single external baseline using the inherited full baseline
# schedule and retain the final checkpoint rather than test-selected best.
if isinstance(cfg.train_cfg, dict):
    max_epochs = int(cfg.train_cfg.get("max_epochs", 300))
    cfg.train_cfg["val_interval"] = max_epochs + 1
    cfg.train_cfg.pop("dynamic_intervals", None)
else:
    max_epochs = 300

if (
    "default_hooks" in cfg
    and "checkpoint" in cfg.default_hooks
):
    cfg.default_hooks["checkpoint"]["save_best"] = None
    cfg.default_hooks["checkpoint"]["interval"] = 50
    cfg.default_hooks["checkpoint"]["max_keep_ckpts"] = 2

cfg.work_dir = "outputs/g00/G00-BL"
cfg.randomness = {
    "seed": 0,
    "deterministic": False,
}

uav_cfg_path = ROOT / "configs/uavdt/rtmdet_tiny_baseline.py"
cfg.dump(str(uav_cfg_path))

# Reload as hard syntax/config check.
check = Config.fromfile(str(uav_cfg_path))
if int(check.model["bbox_head"]["num_classes"]) != 3:
    raise RuntimeError("UAVDT num_classes config failure")

batch_size = int(check.train_dataloader["batch_size"])
drop_last = bool(
    check.train_dataloader.get("drop_last", False)
)

train_frames = split_manifest["train_frames"]
if drop_last:
    steps_per_epoch = train_frames // batch_size
else:
    steps_per_epoch = math.ceil(train_frames / batch_size)

planned_baseline_steps = steps_per_epoch * max_epochs

design = {
    "stage": "G00",
    "status": "DESIGN_FROZEN_BEFORE_FORMAL_G00_RESULTS",
    "dataset": "UAVDT",
    "official_split": {
        "train_sequences": 30,
        "test_sequences": 20,
        "train_frames": split_manifest["train_frames"],
        "test_frames": split_manifest["test_frames"],
    },
    "information_boundary": {
        "test_is_selection_set": False,
        "checkpoint_selection":
            "final checkpoint only; no UAVDT test-based selection",
        "single_seed": 0,
    },
    "comparison_set": [
        {
            "experiment_id": "G00-BL",
            "method": "RTMDet-tiny fixed UAVDT baseline",
            "initialization":
                "same public-pretraining mechanism inherited from "
                "the frozen VisDrone baseline configuration",
            "config":
                "configs/uavdt/rtmdet_tiny_baseline.py",
            "epochs": max_epochs,
            "planned_optimizer_steps":
                planned_baseline_steps,
            "seed": 0,
        },
        {
            "experiment_id": "G00-PEFT",
            "method": "Conv-Adapter",
            "definition":
                "frozen P00-04 Residual Parallel Conv-Adapter "
                "design; gamma=4; adapter-only",
            "source_definition":
                "experiments/manifests/p00_formal/P00-04.json",
            "initialization":
                "fresh frozen G00-BL final checkpoint",
            "optimizer_steps": 42525,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "sampler": "uniform_full_train",
            "seed": 0,
            "no_hyperparameter_search": True,
        },
        {
            "experiment_id": "G00-S2R",
            "method": "frozen S2R-Det / K-star configuration",
            "definition": {
                "layers": [
                    "bbox_head.cls_convs.0.1.conv",
                    "bbox_head.reg_convs.0.1.conv",
                ],
                "rank": 8,
                "update_scope": "FLCR-only",
                "sampler": "uniform_full_train",
                "retention_lambda": 0.0,
                "optimizer_steps": 42525,
                "lr": 0.0003,
                "weight_decay": 0.0001,
                "seed": 0,
            },
            "initialization":
                "fresh frozen G00-BL final checkpoint",
            "no_location_search": True,
            "no_rank_search": True,
            "no_sampler_search": True,
            "no_lambda_search": True,
        },
    ],
    "evaluation": {
        "primary":
            "official UAVDT detection evaluator AP@0.5",
        "diagnostic": [
            "COCO-style metrics",
            "weather",
            "altitude",
            "viewpoint multi-label",
            "occlusion",
            "out-of-view",
            "vehicle category",
        ],
    },
    "interpretation":
        "single-seed limited cross-dataset reproduction only",
}

analysis_dir = ROOT / "analysis/g00"
analysis_dir.mkdir(parents=True, exist_ok=True)
(analysis_dir / "g00_design_freeze.json").write_text(
    json.dumps(design, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# Three preregistered experiment manifests.
manifest_out = ROOT / "experiments/manifests/g00"
manifest_out.mkdir(parents=True, exist_ok=True)

g00_manifests = {
    "G00-BL": {
        "experiment_id": "G00-BL",
        "stage": "G00",
        "execution": "FORMAL_NEW",
        "method": "baseline",
        "config": "configs/uavdt/rtmdet_tiny_baseline.py",
        "seed": 0,
        "checkpoint_selection": "final_only",
        "test_adaptive_use": False,
    },
    "G00-PEFT": {
        "experiment_id": "G00-PEFT",
        "stage": "G00",
        "execution": "FORMAL_NEW",
        "method": "conv_adapter",
        "source_definition":
            "experiments/manifests/p00_formal/P00-04.json",
        "baseline_checkpoint":
            "PENDING_G00_BL_FINAL_CHECKPOINT",
        "steps": 42525,
        "seed": 0,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "sampler": "uniform_full_train",
        "test_adaptive_use": False,
    },
    "G00-S2R": {
        "experiment_id": "G00-S2R",
        "stage": "G00",
        "execution": "FORMAL_NEW",
        "method": "frozen_kstar",
        "baseline_checkpoint":
            "PENDING_G00_BL_FINAL_CHECKPOINT",
        "layers": [
            "bbox_head.cls_convs.0.1.conv",
            "bbox_head.reg_convs.0.1.conv",
        ],
        "rank": 8,
        "steps": 42525,
        "seed": 0,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "sampler": "uniform_full_train",
        "retention_lambda": 0.0,
        "test_adaptive_use": False,
    },
}

for eid, obj in g00_manifests.items():
    (manifest_out / f"{eid}.json").write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

# Registry pre-registration.
exp_path = ROOT / "experiments/registry/experiments.csv"
with exp_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    exp_fields = reader.fieldnames
    exp_rows = list(reader)

existing = {r["experiment_id"] for r in exp_rows}
dataset_sha = sha256(
    ROOT / "data/manifests/uavdt_coco_conversion.json"
)
config_sha = sha256(uav_cfg_path)

new_exp = [
    {
        "experiment_id": "G00-BL",
        "stage": "G00",
        "hypothesis":
            "Establish one fixed RTMDet-tiny UAVDT external baseline.",
        "git_commit": "PENDING_G00_FREEZE_COMMIT",
        "config_sha": config_sha,
        "dataset_manifest_sha": dataset_sha,
        "checkpoint_sha": "PUBLIC_PRETRAIN_INHERITED",
        "trainable_scope": "full baseline training",
        "insertion_layers": "",
        "rank": "",
        "sampler": "official baseline pipeline",
        "retention_lambda": "",
        "seed": "0",
        "budget_tier": "EXTERNAL_BASELINE",
        "expected_steps": str(planned_baseline_steps),
        "wall_time_cap": "36h",
        "status": "PLANNED",
        "valid": "",
        "actual_steps": "",
        "wall_time": "",
        "peak_vram": "",
        "ap": "",
        "aps": "",
        "arsmall": "",
        "vt16": "",
        "notes":
            "official_sequence_split=true;"
            "uavdt_train_sequences=30;"
            "uavdt_test_sequences=20;"
            "test_selection=false;"
            "final_checkpoint_only=true;"
            "single_seed_limited_reproduction=true",
    },
    {
        "experiment_id": "G00-PEFT",
        "stage": "G00",
        "hypothesis":
            "Frozen strong generic Conv-Adapter PEFT transfers to UAVDT.",
        "git_commit": "PENDING_G00_FREEZE_COMMIT",
        "config_sha": config_sha,
        "dataset_manifest_sha": dataset_sha,
        "checkpoint_sha": "PENDING_G00_BL",
        "trainable_scope":
            "Conv-Adapter only; frozen P00-04 definition",
        "insertion_layers": "4 CSPNeXt blocks",
        "rank": "",
        "sampler": "uniform_full_train",
        "retention_lambda": "0.0",
        "seed": "0",
        "budget_tier": "L",
        "expected_steps": "42525",
        "wall_time_cap": "8h",
        "status": "PLANNED_AFTER_G00_BL",
        "valid": "",
        "actual_steps": "",
        "wall_time": "",
        "peak_vram": "",
        "ap": "",
        "aps": "",
        "arsmall": "",
        "vt16": "",
        "notes":
            "gamma=4;no_search=true;"
            "single_seed_limited_reproduction=true",
    },
    {
        "experiment_id": "G00-S2R",
        "stage": "G00",
        "hypothesis":
            "Frozen VisDrone K-star definition transfers to UAVDT.",
        "git_commit": "PENDING_G00_FREEZE_COMMIT",
        "config_sha": config_sha,
        "dataset_manifest_sha": dataset_sha,
        "checkpoint_sha": "PENDING_G00_BL",
        "trainable_scope": "FLCR Head rank8 only",
        "insertion_layers":
            "bbox_head.cls_convs.0.1.conv;"
            "bbox_head.reg_convs.0.1.conv",
        "rank": "8",
        "sampler": "uniform_full_train",
        "retention_lambda": "0.0",
        "seed": "0",
        "budget_tier": "L",
        "expected_steps": "42525",
        "wall_time_cap": "8h",
        "status": "PLANNED_AFTER_G00_BL",
        "valid": "",
        "actual_steps": "",
        "wall_time": "",
        "peak_vram": "",
        "ap": "",
        "aps": "",
        "arsmall": "",
        "vt16": "",
        "notes":
            "frozen_kstar=true;no_location_search=true;"
            "no_rank_search=true;no_sampler_search=true;"
            "no_lambda_search=true;"
            "single_seed_limited_reproduction=true",
    },
]

for row in new_exp:
    if row["experiment_id"] not in existing:
        exp_rows.append(row)

with exp_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=exp_fields,
        lineterminator="\n",
    )
    w.writeheader()
    w.writerows(exp_rows)

budget_path = ROOT / "experiments/registry/budget.csv"
with budget_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    budget_fields = reader.fieldnames
    budget_rows = list(reader)

existing_budget = {
    r["experiment_id"] for r in budget_rows
}

budget_new = [
    {
        "stage": "G00",
        "experiment_id": "G00-BL",
        "budget_tier": "EXTERNAL_BASELINE",
        "planned_steps": str(planned_baseline_steps),
        "planned_wall_time": "36h",
        "actual_steps": "",
        "actual_wall_time": "",
        "gpu_hours": "",
        "status": "PLANNED",
        "notes":
            "single external baseline; 300 inherited epochs; "
            "official train sequences only; final checkpoint only",
    },
    {
        "stage": "G00",
        "experiment_id": "G00-PEFT",
        "budget_tier": "L",
        "planned_steps": "42525",
        "planned_wall_time": "8h",
        "actual_steps": "",
        "actual_wall_time": "",
        "gpu_hours": "",
        "status": "PLANNED_AFTER_G00_BL",
        "notes":
            "Conv-Adapter frozen definition; seed0; no search",
    },
    {
        "stage": "G00",
        "experiment_id": "G00-S2R",
        "budget_tier": "L",
        "planned_steps": "42525",
        "planned_wall_time": "8h",
        "actual_steps": "",
        "actual_wall_time": "",
        "gpu_hours": "",
        "status": "PLANNED_AFTER_G00_BL",
        "notes":
            "frozen K-star; seed0; no search",
    },
]

for row in budget_new:
    if row["experiment_id"] not in existing_budget:
        budget_rows.append(row)

with budget_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=budget_fields,
        lineterminator="\n",
    )
    w.writeheader()
    w.writerows(budget_rows)

print("PREPARE_UAVDT_G00=PASS")
print("TRAIN_SEQUENCES=", len(train_seqs))
print("TEST_SEQUENCES=", len(test_seqs))
print("TRAIN_FRAMES=", split_manifest["train_frames"])
print("TEST_FRAMES=", split_manifest["test_frames"])
print("TRAIN_ANNOTATIONS=", split_outputs["train"]["annotations"])
print("TEST_ANNOTATIONS=", split_outputs["test"]["annotations"])
print("BASELINE_BATCH_SIZE=", batch_size)
print("BASELINE_MAX_EPOCHS=", max_epochs)
print("BASELINE_PLANNED_STEPS=", planned_baseline_steps)
print("G00_METHODS=G00-BL,G00-PEFT,G00-S2R")
