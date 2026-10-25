# S²R-Det — Preliminary Reference Implementation

Paper: **S²R-Det: Failure-Contrastive Sensitivity-Guided Repair for Lightweight UAV Small-Object Detection**  
Hongyuan Zhang, Yamin Cheng, and Hongwei Duan · Xihua University

S²R-Det repairs a trained detector on its original task using labeled images and a
restricted adaptation budget. It freezes the original detector weights, places
compact repair branches using failure–success sensitivity contrasts, and balances
failure exposure with preservation of selected baseline predictions.

**Workflow:** Failure Atlas → Failure-Contrastive Sensitivity Localization (FCSL)
→ Fusible Low-Rank Convolutional Repair (FLCR) → Failure-Stratified Sampling /
Prediction Retention → Fusion → Evaluation.

## Repository structure

| Path | Purpose |
|---|---|
| `s2r_det/failure_atlas/`, `sensitivity/` | Failure attribution, diagnostic sensitivities, semantic placement |
| `s2r_det/adapters/`, `sampling/`, `retention/` | Repair branches, image streams, fixed-teacher targets and losses |
| `s2r_det/training/`, `integrations/` | Optimization protocols and native detector interfaces |
| `s2r_det/data/`, `evaluation/`, `deployment/` | Annotation conversion, metrics, statistics and fusion checks |
| `configs/` | Shared repair settings, comparison recipes and detector templates |
| `scripts/`, `tests/` | Command wrappers and synthetic CPU tests |
| `docs/` | [Methodology](docs/methodology.md), [usage](docs/usage.md), [paper–code map](docs/paper-code-map.md) |

## Installation

Python 3.9 or later is required. From the repository root in a dedicated environment:

```bash
python -m pip install '.[tensor,evaluation,test]'
python -m s2r_det --help
python -m pytest -q -p no:cacheprovider
```

Core dependencies are NumPy and Pillow; the selected extras add PyTorch,
pycocotools and pytest. Native RTMDet execution additionally needs compatible
MMDetection 3.3.0, MMEngine and compiled MMCV. The `detection` extra specifies
those versions, but a platform-compatible Torch/CUDA/MMCV installation must be
prepared separately. `environment.yml` records the paper's environment and is
not a validated installation lock. ONNX Runtime and TensorRT are optional.

## Quick start

The commands below operate on **user-supplied local data and trained weights**.
Set `S2R_SPLIT_ROOT` to a VisDrone split containing `images/` and `annotations/`;
set `S2R_TRAIN_IMAGES` and `S2R_TEST_IMAGES` to their image directories.
`S2R_CONFIG` is a reviewed detector config, `S2R_BASELINE` its compatible baseline
checkpoint, `S2R_INSERTION_SETS` the declared convolution-path sets, and
`S2R_PARAMETER_BUDGET` the architecture-specific repair capacity.
See [usage](docs/usage.md) for these input contracts.

```bash
python -m s2r_det convert-visdrone --split-root "$S2R_SPLIT_ROOT" --output outputs/train.json
python -m s2r_det infer --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/train.json --image-root "$S2R_TRAIN_IMAGES" --output outputs/baseline-train.json
python -m s2r_det atlas --predictions outputs/baseline-train.json --split train --fit-training-quartiles --output outputs/atlas.json
python -m s2r_det manifest --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/train.json --image-root "$S2R_TRAIN_IMAGES" --insertion-sets "$S2R_INSERTION_SETS" --output outputs/manifest.json
python -m s2r_det sensitivity --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/train.json --image-root "$S2R_TRAIN_IMAGES" --atlas outputs/atlas.json --manifest outputs/manifest.json --budget "$S2R_PARAMETER_BUDGET" --output outputs/selection.json
python -m s2r_det cache --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/train.json --image-root "$S2R_TRAIN_IMAGES" --atlas outputs/atlas.json --output outputs/teacher-cache.json
python -m s2r_det train --recipe configs/repair/s2r_det.json --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/train.json --image-root "$S2R_TRAIN_IMAGES" --atlas outputs/atlas.json --manifest outputs/manifest.json --selection outputs/selection.json --retention-cache outputs/teacher-cache.json --seed 0 --output outputs/repair-seed0
```

Run repair separately with seeds 0, 1 and 2. Successful FLCR training writes
`repair-final.pt`, a fused `deployment-final.pt`, and metadata to the chosen output
directory. It checks corresponding pre-NMS outputs on one supplied image before
writing the deployment checkpoint. This local check does not establish AP equality.
Synthetic fusion and shared-parameter checks can be run without a detector:

```bash
python -m pytest -q tests/unit/test_scientific_invariants.py -k 'zero_init_freeze_shapes_and_fusion or shared_head_count_and_fused_aliases' -p no:cacheprovider
```

Convert the test split separately to `outputs/test.json`, then evaluate each model:

```bash
python -m s2r_det infer --detector-config "$S2R_CONFIG" --checkpoint outputs/repair-seed0/deployment-final.pt --annotations outputs/test.json --image-root "$S2R_TEST_IMAGES" --output outputs/repaired-test.json
python -m s2r_det evaluate --annotations outputs/test.json --predictions outputs/repaired-test.json --output outputs/test-metrics.json
python -m s2r_det infer --detector-config "$S2R_CONFIG" --checkpoint "$S2R_BASELINE" --annotations outputs/test.json --image-root "$S2R_TEST_IMAGES" --output outputs/baseline-test.json
python -m s2r_det diagnostics --baseline outputs/baseline-test.json --repaired outputs/repaired-test.json --output outputs/gt-transitions.json
```

The installed `s2r-det` command and wrappers under `scripts/` dispatch the same
interfaces. `python -m s2r_det <command> --help` lists the arguments.

## Configuration and supported scope

Repair recipes in `configs/repair/` and `configs/experiments/` inherit explicit
shared settings through relative `extends` paths. `configs/experiments/index.json`
lists the distinct comparison conditions. The generic repair recipe supports
architecture-specific capacity; the tiny-primary recipe additionally checks the
conditional 3,216-parameter configuration.

The four `configs/rtmdet/` model definitions are **templates, not authenticated
historical experiment configurations**. Supply compatible baselines, actual module
manifests and feasible insertion sets. FCSL does not default to Head.

The CPU scientific primitives and synthetic workflows are tested. Native RTMDet
integration is implemented but has not been run against real detector checkpoints.
YOLOv8n requires a user-supplied native adapter; its Bottleneck comparison and the
Patch-style comparison are not implemented. UAVDT utilities distinguish class-aware
COCO AP, pooled class-agnostic AP50/AP70 and equal-weight sequence AP70; the
class-agnostic interpolation policy is an explicit implementation choice.
ONNX Runtime is dependency-gated, and TensorRT requires an external engine adapter;
neither backend has runtime validation here.

## Reproducibility and availability

This is a preliminary paper-aligned reference implementation. Original experimental
weights and logs are not included, and the paper's results have not been independently
reproduced with this reconstructed package. The author plans to release the final
experimental source after manuscript acceptance and may revise this preview after
additional validation.

## Citation, contact and license

Use the paper title and authors above when referring to this work. A DOI, acceptance
status and publication date are not yet specified. Corresponding author:
Yamin Cheng, `chengyamin16@gmail.com`.

A software license has not yet been selected; see [LICENSE](LICENSE) and
[dependency notices](NOTICE.md). The author must resolve licensing before public upload.
