"""MMDetection 3.3 scale template inheriting the tiny ten-class loss/inference policy.

Requires a compatible local baseline; historical configuration equivalence is
unverified. No pretrained initialization or asset download is configured.
"""

_base_ = "./rtmdet_tiny_visdrone.py"
model = {
    "backbone": {"deepen_factor": 0.33, "widen_factor": 0.5},
    "neck": {"in_channels": [128, 256, 512], "out_channels": 128, "num_csp_blocks": 1},
    "bbox_head": {"in_channels": 128, "feat_channels": 128, "exp_on_reg": False},
}
