# -*- coding: utf-8 -*-
"""Build multi-level comparison table for 4.6 (v2, expanded literature).
Our methods = measured real runs; literature = cited from papers (source column).
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

OUT = REPO / 'main_results'

ct = pd.read_csv(OUT / 'complexity_table.csv')
perf = pd.read_csv(OUT / 'panel_481a_pareto_latency.csv')
perf_a = perf[perf['dataset'] == 'A_UCI_Drift'].set_index('method')['macro_f1'].to_dict()

rows = []

# ---------- ours (dataset A, measured) ----------
for meth in ['base', 'cal_sup', 'mixup_std', 'cg_sup']:
    sub = ct[(ct['dataset'] == 'UCI_Drift') & (ct['method'] == meth)].iloc[0]
    rows.append({
        'method': f'Ours-{meth}', 'source': 'this work (measured)',
        'dataset': 'UCI Drift (A)', 'protocol': 'LOBO, budget=0.10, 5 seeds',
        'performance_metric': 'macro-F1', 'performance_value': round(perf_a.get(meth, np.nan), 4),
        'params': int(sub['params']), 'macs': int(sub['macs']),
        'inference_ms': round(float(sub['inference_ms']), 4),
        'storage_kb_int8': float(sub['weight_storage_kb_int8']),
        'has_overhead': True})

# ---------- literature with overhead (cited) ----------
lit_overhead = [
    dict(method='StrongDANN+ST1',
         source='Gao & Yang, IEEE SCECS 2026, DOI 10.1109/SCECS69812.2026.11566053',
         dataset='UCI Drift (36-month)', protocol='unsup DA + dynamic-threshold self-training, full test',
         performance_metric='accuracy', performance_value=0.8114,
         params=250_000, macs=0.49e6, inference_ms=0.5, storage_kb_int8=np.nan),
    dict(method='LiteSensor-Net',
         source='Kim et al., Molecules 2026, DOI 10.3390/molecules31111884',
         dataset='UCI Drift (36-month)', protocol='chronological forward-chaining + KD-DM-20',
         performance_metric='accuracy / macro-F1', performance_value=0.9263,
         params=np.nan, macs=np.nan, inference_ms=6.3, storage_kb_int8=5.99),
    dict(method='MSD-GasNet',
         source='Sensors 26(15):4810, 2026, DOI 10.3390/s26154810',
         dataset='self-built VOC (5 gases)', protocol='5-class, own dataset (NOT UCI)',
         performance_metric='accuracy', performance_value=0.9680,
         params=796_645, macs=np.nan, inference_ms=2.537, storage_kb_int8=np.nan),
]
for d in lit_overhead:
    rows.append({**d, 'has_overhead': True})

# ---------- literature performance-only (cited, no overhead) ----------
lit_perf = [
    dict(method='CDAN+SAM',
         source='PMC10892276 (Sensors & Materials), 2024',
         dataset='UCI Drift (A)', protocol='batch1 source -> batch K target, avg over K=2..10',
         performance_metric='accuracy', performance_value=0.9032,
         params=np.nan, macs=np.nan, inference_ms=np.nan, storage_kb_int8=np.nan),
    dict(method='CDAN',
         source='PMC10892276 (Sensors & Materials), 2024',
         dataset='UCI Drift (A)', protocol='batch1 source -> batch K target, avg over K=2..10',
         performance_metric='accuracy', performance_value=0.8653,
         params=np.nan, macs=np.nan, inference_ms=np.nan, storage_kb_int8=np.nan),
    dict(method='DANN+SAM',
         source='PMC10892276 (Sensors & Materials), 2024',
         dataset='UCI Drift (A)', protocol='batch1 source -> batch K target, avg over K=2..10',
         performance_metric='accuracy', performance_value=0.8078,
         params=np.nan, macs=np.nan, inference_ms=np.nan, storage_kb_int8=np.nan),
    dict(method='AMDS-PFFA',
         source='Zhang et al., IEEE TIM 2025',
         dataset='UCI Drift (36-month)', protocol='unsup multi-source attention DA',
         performance_metric='accuracy', performance_value=0.8320,
         params=np.nan, macs=np.nan, inference_ms=np.nan, storage_kb_int8=np.nan),
    dict(method='SDRMixer',
         source='Sensors 2026, 14(7):161, DOI 10.3390/s14070161',
         dataset='own sensor-array calibration (mixed gas)', protocol='concentration regression (NOT classification)',
         performance_metric='R2', performance_value=0.9872,
         params=48_700, macs=1.0e6, inference_ms=np.nan, storage_kb_int8=np.nan),
]
for d in lit_perf:
    rows.append({**d, 'has_overhead': False})

t = pd.DataFrame(rows)
t.to_csv(OUT / 'table_46_multilevel_comparison.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)
print(t.to_string(index=False))
print()
print(f'saved -> main_results/table_46_multilevel_comparison.csv  ({len(t)} rows)')
