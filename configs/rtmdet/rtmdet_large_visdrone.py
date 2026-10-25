"""MMDetection 3.3 scale template inheriting the tiny ten-class loss/inference policy.

Requires a compatible local baseline; historical configuration equivalence is
unverified. No pretrained initialization or asset download is configured.
"""

_base_ = "./rtmdet_tiny_visdrone.py"
model = {
    "backbone": {"deepen_factor": 1.0, "widen_factor": 1.0},
    "neck": {"in_channels": [256, 512, 1024], "out_channels": 256, "num_csp_blocks": 3},
    "bbox_head": {"in_channels": 256, "feat_channels": 256, "exp_on_reg": True},
}
