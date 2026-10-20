# -*- coding: utf-8 -*-
"""Measure ResNet34 real overhead on CDSL 6-dim input (the only heavy model that forwards)."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import ast
import torch
import torch.nn as nn
import torchvision.models as tvm
import thop
import time

src = open(REPO / 'overhead' / 'measure_reproduced_overhead.py', encoding='utf-8').read()
tree = ast.parse(src)
funcs = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
            'build_resnet34', 'convert_2d_to_1d', '_conv1d_like', '_pool1d_like',
            '_avgpool1d_like', '_replace_last_linear', '_replace_first_linear'):
        funcs[node.name] = ast.get_source_segment(src, node)
ns = {'tvm': tvm, 'nn': nn, 'NUM_CLASSES': 6}
for fname, fsrc in funcs.items():
    exec(compile(ast.parse(fsrc), '<func>', 'exec'), ns)

m = ns['build_resnet34']()
m.eval()
x = torch.randn(1, 1, 6)
with torch.no_grad():
    m(x)
macs, params = thop.profile(m, inputs=(x,), verbose=False)
print('ResNet34 @ CDSL 6-dim:')
print(f'  params    = {params:.0f}  ({params/1e6:.4f} M)')
print(f'  macs      = {macs:.0f}  ({macs/1e6:.4f} M)')
print(f'  flops     = {macs*2:.0f}  ({macs*2/1e6:.4f} M)')

# inference latency (same protocol as CGDA measurement: single forward, warmed up)
with torch.no_grad():
    for _ in range(5):
        m(x)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        m(x)
    dt = (time.perf_counter() - t0) / n * 1000
print(f'  inference = {dt:.6f} ms')
