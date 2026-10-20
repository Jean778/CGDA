# -*- coding: utf-8 -*-
"""Update CDSL deployment table: add params/macs/flops columns + ResNet34 heavy model.
Heavy models VGG13/VGG19/AlexNet/GoogLeNet are structurally not runnable on 6-dim input
(marked N/A with reason, same convention as GasNet/Sniff Multinose).
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
import numpy as np
import pandas as pd

OUT = r'REPO/main_results'
lit = pd.read_csv(os.path.join(OUT, 'table_cdsl_lit_overhead.csv'))
cgda = pd.read_csv(os.path.join(OUT, 'table_cdsl_overhead.csv'))

rows = []

# --- 4 runnable literature light models ---
for _, r in lit.iterrows():
    rows.append({
        'dataset': 'CDSL',
        'direction': r['direction'],
        'model': r['model'],
        'method': 'base',
        'macro_f1': r['macro_f1'],
        'accuracy': r['accuracy'],
        'ece': r['ece'],
        'params': r['params'],
        'params_M': r['params_M'],
        'macs': r['flops'] / 2.0,
        'flops': r['flops'],
        'flops_M': r['flops_M'],
        'peak_rss_MB': r['peak_rss_MB'],
        'inference_ms': r['inference_ms'],
        'error': '',
    })

# --- CGDA 6 methods ---
for _, r in cgda.iterrows():
    flops = r['macs'] * 2
    rows.append({
        'dataset': 'CDSL',
        'direction': r['direction'],
        'model': 'CGDA',
        'method': r['method'],
        'macro_f1': r['macro_f1'],
        'accuracy': r['accuracy'],
        'ece': r['ece'],
        'params': r['params'],
        'params_M': r['params_M'],
        'macs': r['macs'],
        'flops': flops,
        'flops_M': flops / 1e6,
        'peak_rss_MB': r['peak_rss_MB'],
        'inference_ms': r['inference_ms'],
        'error': '',
    })

# --- ResNet34 (heavy, measured on 6-dim input: real numbers) ---
resnet34 = dict(params=7221318, params_M=7221318 / 1e6, macs=7316032,
                flops=14632064, flops_M=14632064 / 1e6, inference_ms=5.566547)
for direction in ['master_slave1', 'master_slave2']:
    rows.append({
        'dataset': 'CDSL', 'direction': direction, 'model': 'ResNet34', 'method': 'base',
        'macro_f1': None, 'accuracy': None, 'ece': None,
        'params': resnet34['params'], 'params_M': resnet34['params_M'],
        'macs': resnet34['macs'], 'flops': resnet34['flops'], 'flops_M': resnet34['flops_M'],
        'peak_rss_MB': None, 'inference_ms': resnet34['inference_ms'],
        'error': 'params/macs/flops measured on 6-dim input; performance not trained (see note)',
    })

# --- structurally not runnable on 6-dim input ---
na_reasons = [
    ('VGG13', 'structural: 3x MaxPool1d reduces 6->3->1->0, max_pool1d output size 0'),
    ('VGG19', 'structural: 3x MaxPool1d reduces 6->3->1->0, max_pool1d output size 0'),
    ('AlexNet', 'structural: first conv kernel 11 > input length 6'),
    ('GoogLeNet', 'structural: max_pool1d output size 0 on 6-dim input'),
    ('GasNet', 'structural: MaxPool1d reduces 6->3->1->0 (needs >=32-dim input)'),
    ('Sniff Multinose', 'structural: n_groups=4 -> seg=1 -> squeeze(1) collapses channel dim'),
]
for direction in ['master_slave1', 'master_slave2']:
    for name, reason in na_reasons:
        rows.append({
            'dataset': 'CDSL', 'direction': direction, 'model': name, 'method': 'base',
            'macro_f1': None, 'accuracy': None, 'ece': None,
            'params': None, 'params_M': None, 'macs': None, 'flops': None, 'flops_M': None,
            'peak_rss_MB': None, 'inference_ms': None, 'error': reason,
        })

d = pd.DataFrame(rows)
order = ['dataset', 'direction', 'model', 'method', 'macro_f1', 'accuracy', 'ece',
         'params', 'params_M', 'macs', 'flops', 'flops_M',
         'peak_rss_MB', 'inference_ms', 'error']
d = d[order]
d.to_csv(os.path.join(OUT, 'table_cdsl_deployment_light.csv'), index=False, float_format='%.6f')
print('saved ->', os.path.join(OUT, 'table_cdsl_deployment_light.csv'), 'rows:', len(d))
pd.set_option('display.width', 280)
pd.set_option('display.max_rows', 100)
print(d[['direction', 'model', 'method', 'params', 'params_M', 'macs', 'flops', 'flops_M', 'error']].to_string(index=False))
