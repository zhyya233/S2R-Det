# S²R-Det

Official paper-release code layout for S²R-Det.

This public package uses descriptive file names rather than internal experiment-stage IDs.
Internal variant identifiers that remain inside the implementation are preserved only where
they are part of the frozen experiment logic or checkpoint compatibility.

## Repository layout

```text
S2R-Det/
├── configs/
│   ├── data/
│   │   └── visdrone2019_det.py
│   ├── visdrone/
│   │   ├── rtmdet_tiny_baseline.py
│   │   ├── s2r_det_adaptation.py
│   │   └── rtmdet_x_baseline.py
│   ├── uavdt/
│   │   └── rtmdet_tiny_baseline.py
│   └── reproducibility/
│       ├── visdrone_rtmdet_tiny_baseline.py
│       ├── visdrone_rtmdet_x_baseline.py
│       └── uavdt_rtmdet_tiny_baseline.py
├── s2r_det/
│   ├── adapters/
│   └── training/
├── scripts/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── analysis/
├── requirements.txt
├── environment.yml
└── weights/
    └── SHA256SUMS
```

## Environment

Create the environment with either:

```bash
conda env create -f environment.yml
conda activate s2r-det
```

or install the recorded Python dependencies with:

```bash
pip install -r requirements.txt
```

The original experiments used Python 3.10, MMDetection 3.3.0,
MMCV 2.1.0, MMEngine 0.10.7, and a CUDA-enabled PyTorch build.

## Dataset preparation

The datasets are not redistributed.

### VisDrone2019-DET

Expected logical layout:

```text
VisDrone2019-DET/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-test-dev/
│   ├── images/
│   └── annotations/
└── coco_annotations/
    ├── visdrone2019_det_train.json
    ├── visdrone2019_det_val.json
    └── visdrone2019_det_testdev.json
```

VisDrone conversion utility:

```bash
python scripts/data/convert_visdrone_to_coco.py --help
```

### UAVDT

Preparation utility:

```bash
python scripts/data/prepare_uavdt.py --help
```

Local dataset paths in configs may need to be adjusted to your machine.

## Main configs

VisDrone RTMDet-Tiny baseline:

```text
configs/visdrone/rtmdet_tiny_baseline.py
```

S²R-Det adaptation recipe:

```text
configs/visdrone/s2r_det_adaptation.py
```

VisDrone RTMDet-X capacity baseline:

```text
configs/visdrone/rtmdet_x_baseline.py
```

UAVDT baseline:

```text
configs/uavdt/rtmdet_tiny_baseline.py
```

## Training

Standard MMDetection training entry point:

```bash
python scripts/train/train_detector.py   configs/visdrone/rtmdet_tiny_baseline.py
```

S²R-Det training entry point:

```bash
python scripts/train/train_s2r_det.py --help
```

The S²R-Det training entry point exposes the frozen adaptation controls explicitly;
use `--help` to inspect the required checkpoint/control arguments.

## Evaluation

Baseline evaluation:

```bash
python scripts/eval/evaluate_baseline.py --help
```

S²R-Det evaluation:

```bash
python scripts/eval/evaluate_s2r_det.py --help
```

UAVDT evaluation:

```bash
python scripts/eval/evaluate_uavdt.py --help
```

VisDrone-C evaluation:

```bash
python scripts/eval/evaluate_visdrone_c.py --help
```

Paired bootstrap analysis:

```bash
python scripts/eval/paired_bootstrap.py --help
```

Deployment/fusion evaluation:

```bash
python scripts/eval/evaluate_deployment.py --help
```

## Analysis utilities

Failure Atlas:

```bash
python scripts/analysis/build_failure_atlas.py --help
```

FCSL sensitivity:

```bash
python scripts/analysis/analyze_fcsl_sensitivity.py --help
```

Small-vs-all sensitivity:

```bash
python scripts/analysis/compare_small_vs_all_sensitivity.py --help
```

Retention cache:

```bash
python scripts/analysis/build_retention_cache.py --help
```

## Checkpoints

The code-only package does not embed `.pth` files. Checkpoint hashes are retained in:

```text
weights/SHA256SUMS
```

For GitHub, store large checkpoints with Git LFS or attach them to a GitHub Release.

## Naming policy

Public file names describe their function or dataset. Internal stage labels such as
B00/K00/G00/P00/X00/Y00 are intentionally not used in public paths or file names.
Where such identifiers remain inside source code, they are preserved only as frozen
experiment/variant identifiers to avoid altering experimental semantics.
