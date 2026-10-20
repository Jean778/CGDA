# -*- coding: utf-8 -*-
"""Merge reproduced (re-measured) overhead values with literature-reported values
into one final comparison table for 4.6, keyed by dataset (input dim):
A=UCI Drift(128)  B=CDSL(6)  C=GSA(70).
Reproduced models: measured here with thop on 1D input (single-variable).
Literature-only models: values as reported in their papers (input dim as used there).
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

p = REPO / 'main_results'
rep = pd.read_csv(p / 'table_46_reproduced_overhead.csv')
ours = pd.read_csv(p / 'table_46_three_indicators_ours.csv')
lit = pd.read_csv(p / 'table_46_literature_per_dataset.csv')

# dataset -> input dim mapping (ours table uses dataset names)
dim_map = {'UCI_Drift': 128, 'CDSL': 6, 'GSA': 70}

# ---- reproduced models: one row per (model, input_dim) ----
rep_rows = []
for _, r in rep.iterrows():
    rep_rows.append(dict(
        dataset_dim=int(r['input_dim']),
        method=r['model'],
        source='reproduced (this work, thop, 1D input, {}-dim)'.format(int(r['input_dim'])),
        params_M=r['params_M'], flops_M=r['flops_M'],
        params=int(r['params']), flops=int(r['flops']),
        measurement='re-measured', note='1D architecture reproduced from paper/torchvision; input dim {}'.format(int(r['input_dim'])),
    ))

# ---- our methods: use measured params/flops per dataset ----
ours_rows = []
for _, r in ours.iterrows():
    dim = dim_map.get(r['dataset'])
    if dim is None:
        continue
    ours_rows.append(dict(
        dataset_dim=dim,
        method='Ours-{}'.format(r['method']),
        source='this work (measured, EPOCHS=50)',
        params_M=r['params_M'], flops_M=r['flops_M'],
        params=int(r['params_M'] * 1e6), flops=int(r['flops_M'] * 1e6),
        measurement='re-measured', note='our model, input dim {}'.format(dim),
    ))

# ---- literature-only (no open code): reported values ----
lit_rows = []
for _, r in lit.iterrows():
    if pd.notna(r.get('params_M')) and pd.notna(r.get('flops_M')):
        lit_rows.append(dict(
            dataset_dim=128,  # those papers mostly used UCI Drift (A)
            method=str(r['method']),
            source=str(r['source']),
            params_M=r['params_M'], flops_M=r['flops_M'],
            params=int(r['params_M'] * 1e6), flops=int(r['flops_M'] * 1e6),
            measurement='literature-reported', note=str(r.get('note', '')),
        ))

merged = pd.DataFrame(rep_rows + ours_rows + lit_rows)
merged.to_csv(p / 'table_46_final_overhead_merged.csv', index=False)
print('total rows:', len(merged))
print()
pd.set_option('display.width', 260)
print(merged[['dataset_dim', 'method', 'params_M', 'flops_M', 'measurement']].to_string(index=False,
    formatters={'params_M': '{:.4f}'.format, 'flops_M': '{:.4f}'.format}))
