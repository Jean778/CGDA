# -*- coding: utf-8 -*-
"""Merge Table IX model values into the per-dataset literature comparison table.
Source: OTTA-DriftNet (IEEE TIM 2025) Table IX and MLAEC-Net (IEEE Sensors J 2024) Table IX.
Values are as-reported in those papers; input-dimension caveat noted.
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

p = REPO / 'main_results'
t = pd.read_csv(p / 'table_46_literature_per_dataset.csv')

# ensure M-unit columns exist
if 'params_M' not in t.columns:
    t['params_M'] = t['params'] / 1e6
    t['flops_M'] = t['macs'] / 1e6

# Table IX rows (both papers, UCI Drift domain; values as reported)
rows = [
    # OTTA-DriftNet Table IX
    dict(dataset='A (UCI Drift)', method='ODCNN', params_M=0.265766, flops_M=0.212848,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='VGG13', params_M=9.604422, flops_M=25.843610,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='GasNet', params_M=0.382118, flops_M=1.222308,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='TDACNN', params_M=1.362544, flops_M=2.568145,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='M2FL-CCC', params_M=0.745586, flops_M=1.098821,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='SniffConv', params_M=0.183154, flops_M=6.055386,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='SniffMultinose', params_M=0.219486, flops_M=6.682468,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='ODAELM', params_M=0.167239, flops_M=5.869713,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='OTTA-DriftNet', params_M=0.446973, flops_M=5.137291,
         source='OTTA-DriftNet Table IX (Liang et al., IEEE TIM 74, 2025)',
         note='value from Table IX; input dim as used in that paper'),
    # MLAEC-Net Table IX
    dict(dataset='A (UCI Drift)', method='LSTM', params_M=0.187862, flops_M=0.230656,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='VGG19', params_M=20.032690, flops_M=45.967070,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='LeNet', params_M=0.258446, flops_M=4.284246,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='AlexNet', params_M=5.734214, flops_M=19.399150,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='GoogLeNet', params_M=2.979702, flops_M=9.489656,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='ResNet34', params_M=23.850750, flops_M=28.636210,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
    dict(dataset='A (UCI Drift)', method='MLAEC-Net', params_M=0.568672, flops_M=1.108704,
         source='MLAEC-Net Table IX (Liang et al., IEEE Sensors J. 24(16), 2024)',
         note='value from Table IX; input dim as used in that paper'),
]

new = pd.DataFrame(rows)
# keep old columns consistent
for c in ['params', 'macs', 'latency_ms', 'storage_kb', 'performance', 'performance_metric', 'task', 'platform']:
    if c not in new.columns:
        new[c] = np.nan
new['params'] = new['params_M'] * 1e6
new['macs'] = new['flops_M'] / 2 * 1e6
new['has_overhead'] = True

keep_cols = ['dataset', 'method', 'task', 'source', 'performance_metric', 'performance',
             'params', 'macs', 'params_M', 'flops_M', 'latency_ms', 'storage_kb',
             'platform', 'note', 'has_overhead']
for c in keep_cols:
    if c not in t.columns:
        t[c] = np.nan
    if c not in new.columns:
        new[c] = np.nan

t2 = pd.concat([t[keep_cols], new[keep_cols]], ignore_index=True)
t2.to_csv(p / 'table_46_literature_per_dataset.csv', index=False)
print('total rows:', len(t2))
print()
print('methods with params_M+flops_M (usable in Pareto):',
      t2.loc[t2['params_M'].notna() & t2['flops_M'].notna(), 'method'].tolist())
