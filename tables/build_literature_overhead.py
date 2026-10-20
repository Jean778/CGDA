# -*- coding: utf-8 -*-
"""Build per-dataset literature overhead comparison for 4.6.
Literature numbers are quoted from the cited papers (source column). Missing = not reported.
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

OUT = REPO / 'main_results'

rows = []

# ============ Dataset A: UCI Drift (Gas Sensor Array Drift) ============
rows += [
    dict(dataset='A (UCI Drift)', method='StrongDANN+ST1', task='6-class gas ID',
         source='Gao & Yang, IEEE SCECS 2026, DOI 10.1109/SCECS69812.2026.11566053',
         performance_metric='accuracy', performance=0.8114,
         params=250_000, macs=0.49e6, latency_ms=0.5, storage_kb=np.nan,
         platform='reported as sub-ms (edge)'),
    dict(dataset='A (UCI Drift)', method='LiteSensor-Net', task='6-class gas ID',
         source='Kim et al., Molecules 2026, DOI 10.3390/molecules31111884',
         performance_metric='accuracy / macro-F1', performance=0.9263,
         params=np.nan, macs=np.nan, latency_ms=6.3, storage_kb=5.99,
         platform='Raspberry Pi 4B (measured); RAM 31.7 kB, energy 0.04 mJ'),
    dict(dataset='A (UCI Drift)', method='CDAELM', task='6-class gas ID',
         source='E3S Web Conf. 2026, DOI 10.1051/e3sconf/2026XX01013',
         performance_metric='relative gain over ELM', performance=0.1050,
         params=np.nan, macs=np.nan, latency_ms=0.0022, storage_kb=np.nan,
         platform='single-sample inference 2.2 us; training 126 ms; closed-form ELM'),
    dict(dataset='A (UCI Drift)', method='Neuro-Bayesian SNN', task='6-class gas ID',
         source='arXiv 2407.04714',
         performance_metric='accuracy (batch i->i+1)', performance=0.88,
         params=32_000, macs=125_254, latency_ms=np.nan, storage_kb=np.nan,
         platform='energy 626.3 nJ/inf (SNN), 1543.4 nJ (ANN); ANN acc 99%'),
    dict(dataset='A (UCI Drift)', method='CDAN+SAM', task='6-class gas ID',
         source='PMC10892276 (2024)',
         performance_metric='accuracy (batch1->K avg)', performance=0.9032,
         params=np.nan, macs=np.nan, latency_ms=np.nan, storage_kb=np.nan,
         platform='not reported'),
    dict(dataset='A (UCI Drift)', method='AMDS-PFFA', task='6-class gas ID',
         source='Zhang et al., IEEE TIM 2025',
         performance_metric='accuracy', performance=0.8320,
         params=np.nan, macs=np.nan, latency_ms=np.nan, storage_kb=np.nan,
         platform='not reported'),
    dict(dataset='A (UCI Drift)', method='MDFE-Net', task='6-class gas ID',
         source='R Discovery (2026)',
         performance_metric='accuracy (long-term drift)', performance=0.9553,
         params=np.nan, macs=np.nan, latency_ms=np.nan, storage_kb=np.nan,
         platform='not reported'),
]

# ============ Dataset D: AirQualityUCI (Air Quality, Italy) ============
rows += [
    dict(dataset='D (AirQualityUCI)', method='XGBoost-tiny', task='CO / NO2 regression',
         source='arXiv 2511.21857',
         performance_metric='R2 (CO/NO2)', performance=np.nan,
         params=np.nan, macs=np.nan, latency_ms=np.nan, storage_kb=np.nan,
         platform='reports inference time / model size / peak RAM; exact values in paper'),
    dict(dataset='D (AirQualityUCI)', method='AirGuardPlus ensemble (RF+GB+LSTM)',
         task='AQI regression',
         source='IJIFR 2026 (AirGuardPlus AI)',
         performance_metric='R2', performance=0.932,
         params=np.nan, macs=np.nan, latency_ms=50.0, storage_kb=np.nan,
         platform='ensemble forward pass 2-20 ms; prediction <50 ms; load 1.8 s'),
]

# ============ Literature with overhead on related e-nose tasks (context) ============
rows += [
    dict(dataset='context (self-built VOC)', method='MSD-GasNet', task='5-class VOC ID',
         source='Sensors 26(15):4810, 2026',
         performance_metric='accuracy', performance=0.9680,
         params=796_645, macs=np.nan, latency_ms=2.537, storage_kb=np.nan,
         platform='single-sample inference 2.54 ms; training 94 s'),
    dict(dataset='context (ESP32 edge AQI)', method='Lightweight GRU (TFLite)',
         task='AQI forecasting',
         source='Lightweight Edge AI Framework (ESP32)',
         performance_metric='R2', performance=0.9402,
         params=8_289, macs=np.nan, latency_ms=120.0, storage_kb=246.0,
         platform='ESP32 120 ms, smartphone 45 ms; memory <50 MB'),
]

t = pd.DataFrame(rows)
t.to_csv(OUT / 'table_46_literature_per_dataset.csv', index=False)
pd.set_option('display.width', 260)
pd.set_option('display.max_colwidth', 55)
print(t.to_string(index=False))
print()
print(f'saved -> main_results/table_46_literature_per_dataset.csv ({len(t)} rows)')
