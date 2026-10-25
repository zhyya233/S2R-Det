# Usage and input contracts

Run commands from the repository root after the installation in the [README](../README.md).
All detector commands require locally supplied images, annotations and trained weights.

## Data and detector inputs

`convert-visdrone` expects `images/` and `annotations/` under one split directory.
It retains score-one, category 1–10, positive-extent boxes, preserving original
row identity and occlusion. Convert training, validation and test splits separately.
`CocoImages` accepts retained non-crowd COCO annotations with unique image and GT
identities; its detection supervision includes all retained objects.

`S2R_CONFIG` and `S2R_BASELINE` in the README name compatible local files. The
RTMDet adapter disables pretrained initialization, loads tensors strictly and
checks native pre/post-NMS outputs against its explicit NMS/cap trace. It supports
hard class-wise NMS. Native MMDetection/MMCV execution has not been validated here.
The four scale templates inherit the tiny ten-class task/loss/inference policy;
review them against each actual baseline rather than assuming historical identity.

Repair artifacts written by this package use `format_version=1`, recipe,
placement and fusion metadata. Unfused FLCR keys remain `base.*`,
`repair_v.weight`, `repair_d.weight` and `repair_u.weight`; shared aliases remain
shared. Arbitrary older repaired checkpoints need a reviewed conversion and are
not silently accepted as native baselines. No resume interface is exposed.

## Manifest and analysis

`S2R_INSERTION_SETS` is a JSON object mapping the names `backbone_shallow`,
`backbone_late`, `neck_p3`, `neck_p4` and `head` to predeclared lists of actual
canonical convolution paths. An empty list makes that group's insertion infeasible;
it is not a basis for manufacturing a preferred selection. Use architecture-aware
paths. `configs/rtmdet/primary_head_example.json` is only the conditional two-branch,
96-channel Head specification, not a complete five-group insertion policy.

`manifest` traces the supplied detector and records canonical paths, aliases,
group, channels, spatial operations and insertion eligibility. `sensitivity`
requires that manifest, the training Atlas and an explicit parameter budget;
insufficient cohorts or unstable/infeasible placement fail. Detector configs,
baseline and annotations must match the analysis artifacts' recorded hashes.
For a validation Atlas, supply frozen training quartiles with `--quartiles`;
only training images may fit quartiles.

`cache` creates detached dense-index teacher entries for retention-eligible images.
`train` verifies cache/Atlas/manifest/selection identities and applies the recipe.
`s2r_det` recipes require an actual FCSL selection; control placements instead
use `--placement` with a JSON `paths` list. Successful FLCR training fuses branches
and performs a one-image pre-NMS tensor check. Broader deployment verification
requires an appropriate input corpus and independent post-NMS/AP evaluation.

## Repair and comparison recipes

`s2r_det.utils.config.load_recipe(path)` resolves `extends` relative to each JSON
file, recursively merges objects and replaces arrays/scalars, including explicit
null values. Cycles and missing parents fail. Saved training metadata contains the
resolved settings. `configs/repair/recipe.schema.json` describes that effective
recipe, rather than the unexpanded inheritance file.

`configs/repair/common.json` centralizes shared defaults. The complete recipe
`s2r_det.json` changes them explicitly; `s2r_det_tiny_primary.json` additionally
checks the conditional primary parameter count. `configs/rtmdet/model_scales.json`
maps all four model sizes to the shared workflow and individual model templates.

| Comparison family | Updates / seeds | Additional input or constraint |
|---|---|---|
| Complete method | 42,525 / 0–2 | Actual FCSL selection and teacher cache |
| Full fine-tuning | 42,525 or 25,245 / 0–2 | All weights/affine, frozen running statistics; equal-time is the paper's recorded budget |
| Placement | 24,300 / 0 | Explicit manual, FCSL or random manifest; joint Neck/Head has a distinct capacity |
| Sampling × retention | 24,300 / 0–2 | Four factorial conditions remain separate |
| Sampling alternatives | 24,300 / 0 | Random subset or Top-loss needs an explicit support record |
| Rank/budget | 12,150 / 24,300 / 42,525; seed 0 | Rank 4/8/16, uniform, no retention |
| Update scope | 24,300 / 0 | FLCR-only, specified normalization subset or Head |
| Calibration | 12,150 / 0 | Validation only, candidate learning rates named in the files |

The experiment index lists the executable recipe examples. Repeat
`placement_random.json` with separately recorded supplied manifests for different
random placements; no random location is generated implicitly. Fixed-spatial
exploration retains the Laplacian option. Some primary-control sampling/retention
assignments are unspecified in the paper; their uniform/no-retention defaults are
labeled choices. Head-only and norm/bias recipes require `--learning-rate`.
FLCR-plus-normalization requires explicit `norm_paths`. Review all such choices
before comparing a new run with the paper.

`sampling-support` accepts an image-ID list and count. Top-loss also requires one
fixed finite score per image and records the score-file identity. It does not
implement an undocumented score-refresh schedule. Patch-style and YOLO Bottleneck
topologies are not supplied; unsupported controls fail explicitly.

## Metrics and resampling

`infer` writes original-coordinate COCO predictions plus per-image GT, candidate
trace and identity hashes. `evaluate` accepts that artifact or COCO prediction rows.
`diagnostics` requires the matched baseline and repair inference artifacts, then
computes GT Recovery, GT Damage, Net Repair and success retention per image.

```bash
python -m s2r_det bootstrap --unit image --annotations outputs/test.json --baseline outputs/baseline-test.json --repaired outputs/repaired-test.json --replicates 10000 --seed 0 --output outputs/image-interval.json
python -m s2r_det aggregate-seeds --records outputs/seed-metrics.json --output outputs/seed-summary.json
```

Seed aggregation expects an object mapping each metric to its independently
trained seed values, such as `{"APsmall": [value0, value1, value2]}` after replacing
the symbolic values with actual numbers. It computes sample SD; it does not
infer missing values or paired-gain SD from marginal SDs.

For UAVDT, `prepare-uavdt` validates already converted COCO annotations with
globally unique frame IDs, sequence-qualified filenames and explicit `sequence_id`.
Raw split/ignore policies and baseline architecture must be supplied.
`evaluate --dataset uavdt` separates class-aware COCO AP, pooled class-agnostic
AP50/AP70 and equal-weight sequence AP70. `bootstrap --unit sequence` requires two
JSON objects mapping identical sequence IDs to the respective per-sequence AP70
values. It does not bootstrap individual video frames as independent images.

## Optional integrations and deployment

`--adapter-factory module:function` loads a user-provided architecture adapter.
For YOLOv8n it must implement the complete `YOLORepairAdapter` contract, including
native loss/diagnostic objectives, decoder, dense correspondence and candidate
trace. Supply its semantic manifest externally; the `manifest` command's automatic
tracing/group rules support RTMDet. No concrete YOLO runtime is included.

`s2r_det.adapters.flcr.fuse_flcr_inplace` performs analytical fusion;
`deployment.validation.compare_outputs` checks corresponding finite tensors.
Neither checks AP equality. `deployment --factory module:function` requires a
bundle with decoded CPU images and a prediction callback, plus synchronization
when needed. The callback must include preprocessing, transfer, inference, decoding,
NMS and final CPU detections. File reading/decoding and engine construction are
outside the timer. P50/P95 summarize samples; FPS uses mean latency.

`deployment.backends.onnx_session` requires ONNX Runtime and a supplied model.
`tensor_rt_engine` requires TensorRT and an explicit engine/buffer factory. These
backends are unverified; no exporter, engine or successful runtime result is supplied.
`degradation --factory module:function` accepts dataset, decoded-image loader and
prediction callbacks, applying all six S7 operator families at three strengths.
Its metric evaluation still requires actual predictions.

## Validation scope

Synthetic CPU tests cover geometry, Atlas matching, FCSL ranking, shared FLCR
fusion/counts, sampling, detached retention, schedule endpoints, GT transitions,
COCO boundaries, paired resampling and a tiny optimization fixture. They do not
validate native detector training, actual eight-worker execution, GPU behavior,
real-data metrics, optional backends or historical result equivalence.
No weights, datasets, private logs or cached predictions are distributed.
