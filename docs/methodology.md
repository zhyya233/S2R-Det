# Methodology

The manuscript's Method, Experimental Setup and Algorithm 1, together with
Supplementary Sections S1–S7, define the scientific workflow.

## Failure Atlas and image sampling

Correct-class GT–prediction pairs are assigned one to one in descending IoU order,
with GT index then prediction index breaking ties. Attribution checks explicit
pre-NMS, post-NMS, suppression and output-cap records in this priority:

1. **F4 — NMS suppression:** a correct-class candidate with IoU ≥ 0.5 is confirmed
   suppressed, without an equivalent post-NMS match. Cap removal alone is insufficient.
2. **F1 — missed:** no correct-class candidate reaches IoU 0.1.
3. **F2 — localization:** best correct-class IoU is in [0.1, 0.5), with a compatible assignment.
4. **F3 — low confidence:** a post-NMS match at IoU ≥ 0.5 has confidence at or below
   its frozen training class-by-scale first quartile.
5. **F5 — retained success:** a compatible post-NMS match excludes F3 and F4.

Unresolved conflicts are ambiguous. Analysis uses the strict union of original
area < 1,024 and resized area < 256, counting each GT identity once. Association
occurs on all retained GT before this scope filter. F3 can still be successful
under the binary IoU rule used for GT Recovery and GT Damage.

Failure images contain F1–F4. Retention-eligible images have F5 > 0 and more F5
than F1–F4 combined; pools can overlap. Ambiguous instances do not enter these
counts. Active failure-type quotas are bounded proportional allocations in
[0.1, 0.4]; infeasible bounds or required empty pools fail. The 70/30 schedule
preallocates `floor(0.7 * steps + 0.5)` failure-stream updates and shuffles them
with the repair seed. Each update draws a full minibatch with replacement from
one stream. **Every retained GT in each selected image remains detector supervision.**

## Failure-Contrastive Sensitivity Localization

FCSL uses five disjoint semantic groups: shallow backbone, late backbone, Neck P3,
Neck P4 and joint classification/regression Head. A canonical manifest deduplicates
shared convolutions, fixes scored modules and declares each group's insertion
subset before ranking. Eligible sites are ordinary groups=1, 3×3 convolutions.

The fixed analysis set has 256 failure-rich and 256 success-rich training images,
allocated proportionally within training-derived density strata. For each GT,
the diagnostic proxy selects the maximum-IoU dense decoded box before filtering
or NMS, breaking ties by dense index. Native classification and box losses use
the annotated class, quality target 1, unit instance weight and averaging factor 1.
This proxy is separate from the detector's full-image training assigner.

For a pre-normalization convolution output A, the normalized sensitivity is
`mean(abs(A * gradient)) / (sqrt(mean(A**2)) + 1e-12)`. Repeated calls of one
module pool all activation elements, including disconnected calls in the
denominator. Group scores take the median of `log(sensitivity + 1e-12)` across
unique eligible modules; even medians average the central pair. The contrast
subtracts the F5 instance median from the failure-instance median.

Observed contrasts determine ranking. The 1,000 bootstrap replicates resample
whole images within fixed cohort-by-density cells, retaining every F/F5 instance
and repeated-image multiplicity. Fixed group order resolves rank ties. Top-1 or
Top-2 frequency ≥ 0.60 is required for the corresponding one- or two-group budget.
Selection uses declared insertion sets and actual parameter costs. A two-group
request requires a feasible pair; unavailable automatic placement fails explicitly.

## Fusible Low-Rank Convolutional Repair

FLCR adds a linear, bias-free `V(1×1) → D(depthwise 3×3) → U(1×1)` branch in
parallel with the frozen convolution. U starts at zero. D inherits compatible
stride, padding, dilation and boundary handling. Each branch has
`rank * (in_channels + out_channels + 9)` trainable parameters.

The fused update is `DeltaW[o,i,u,v] = sum_k U[o,k] * D[k,u,v] * V[k,i]`.
Fusion retains the original bias and following normalization/activation. All
aliases of a shared convolution share one branch and one fused convolution.
Unsupported operators fail; fusion does not apply to the nonlinear Conv-Adapter.

## Prediction Retention and optimization

The fixed teacher cache selects up to 100 valid raw candidate–class entries before
score filtering/NMS, ordered by sigmoid confidence, dense index then class.
Different class entries at one dense location remain separate. Each stores the
full class logit vector and decoded box. Classification uses every valid cached
entry; boxes use a separate one-to-one same-class IoU ≥ 0.5 F5 mask.

Teacher targets are detached. Classification uses Bernoulli KL with T=2 and T²;
boxes use SmoothL1 with transition delta=1 input pixel and relative weight beta=1.
Both reductions pool their respective entries across eligible image occurrences,
with empty sets contributing zero. Retention applies regardless of sampling
stream. The complete recipe uses lambda=0.25 alongside native full-GT detector loss.
Caches bind baseline, annotations, preprocessing, dense convention and Atlas hashes.

Primary repair uses rank 8, seeds 0–2, 42,525 updates, batch 16, eight workers,
FP32 and AdamW. The 1,000-update warm-up and cosine schedule reach the stated
endpoints. Original weights and running statistics remain frozen. Comparison
recipes preserve their distinct budgets/scopes. Calibration uses the final
12,150-update validation APsmall, then AP, then smaller learning rate; evaluation
uses the prescribed final repair update.

## Evaluation and statistical units

COCO AP/size AP use maxDets 100 and original-image area with native inclusive
endpoints. ARsmall@500 and strict resized-area VT16 AP@500 are separate metrics.
GT Recovery, GT Damage and Net Repair use class-consistent IoU-priority matching
at 0.5, distinct from score-ranked AP matching. Success retention divides retained
successes by baseline successes.

Repair-seed summaries use arithmetic means and sample SD. Paired image Bootstrap
recomputes AP and gives repeated images new COCO identities. Paired sequence
Bootstrap instead targets equal-weight sequence-mean AP70. Neither uncertainty
measure substitutes for the other or for repair-seed variation.

## Explicit implementation choices

Pixel preprocessing uses Pillow RGB decoding, bilinear resizing, half-up integer
dimensions and upper-left RGB114 padding; actual resized dimensions determine
box areas. Python RNG sequences and initialization draw order are not certified
identical to the original experiments. Subgroup scale summaries allocate boundary
ties to the lower group; standard COCO AP retains its inclusive ranges.

UAVDT class-agnostic AP uses COCO's 101 recall points; equivalence to the original
evaluator is unverified. Corruption pixels use documented deterministic RNG,
motion-kernel and JPEG choices. Latency warm-up defaults to a configurable 20
iterations, and supported-operation profiling is not a full detector FLOP count.
These choices and the integration limits in [usage](usage.md) must be considered
when interpreting any new run.
