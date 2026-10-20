# -*- coding: utf-8 -*-
"""Compute CGDA network params/FLOPs via thop (same caliber as lit models)."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
import torch
import thop
from revision_core import Network

m = Network(6, 6)
macs, params = thop.profile(m, inputs=(torch.zeros(1, 1, 6),), verbose=False)
print(f'params={params} macs={macs} flops={macs*2} flops_M={(macs*2)/1e6:.4f} params_M={params/1e6:.4f}')
