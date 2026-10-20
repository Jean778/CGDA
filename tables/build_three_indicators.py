# -*- coding: utf-8 -*-
"""Convert three-indicator table to M units (Params in M, FLOPs in M)."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
from pathlib import Path

p = REPO / 'main_results'
ct = pd.read_csv(p / 'complexity_table.csv')
EPOCHS = 50
ct['flops'] = ct['macs'] * 2
ct['avg_time_per_epoch_s'] = ct['training_time_s'] / EPOCHS

out = pd.DataFrame({
    'dataset': ct['dataset'],
    'method': ct['method'],
    'params_M': ct['params'] / 1e6,
    'flops_M': ct['flops'] / 1e6,
    'macs_M': ct['macs'] / 1e6,
    'avg_time_per_epoch_s': ct['avg_time_per_epoch_s'],
    'training_time_s': ct['training_time_s'],
    'inference_ms': ct['inference_ms'],
})
out.to_csv(p / 'table_46_three_indicators_ours.csv', index=False)
pd.set_option('display.width', 200)
print(out.to_string(index=False,
    formatters={'params_M': '{:.4f}'.format, 'flops_M': '{:.4f}'.format,
                'macs_M': '{:.4f}'.format, 'avg_time_per_epoch_s': '{:.4f}'.format,
                'training_time_s': '{:.3f}'.format, 'inference_ms': '{:.4f}'.format}))
print()
print('saved -> main_results/table_46_three_indicators_ours.csv')
