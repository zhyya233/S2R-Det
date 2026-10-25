"""MMDetection 3.3 scale template inheriting the tiny ten-class loss/inference policy.

Requires a compatible local baseline; historical configuration equivalence is
unverified. No pretrained initialization or asset download is configured.
"""

_base_ = "./rtmdet_tiny_visdrone.py"
model = {
    "backbone": {"deepen_factor": 0.67, "widen_factor": 0.75},
    "neck": {"in_channels": [192, 384, 768], "out_channels": 192, "num_csp_blocks": 2},
    "bbox_head": {"in_channels": 192, "feat_channels": 192, "exp_on_reg": True},
}
