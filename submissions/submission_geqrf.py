"""
Reference baseline submission: plain torch.geqrf (cuSOLVER).

This is the competition's own ref_kernel — guaranteed to satisfy the (H, tau)
compact-Householder contract. We submit it in --mode test to validate the
official popcorn-cli grading path end-to-end, and it doubles as the per-shape
runtime bar our custom kernels must beat.
"""

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)
