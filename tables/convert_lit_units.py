# -*- coding: utf-8 -*-
"""Convert literature params/macs to M units."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
from pathlib import Path

p = REPO / 'main_results'
t = pd.read_csv(p / 'table_46_literature_per_dataset.csv')

t['params_M'] = t['params'] / 1e6
t['flops_M'] = t['macs'] / 1e6
# keep original columns for traceability
t.to_csv(p / 'table_46_literature_per_dataset.csv', index=False)
pd.set_option('display.width', 220)
show = t[['dataset', 'method', 'params_M', 'flops_M', 'latency_ms', 'storage_kb']].copy()
print(show.to_string(index=False,
    formatters={'params_M': lambda v: '{:.4f}'.format(v) if pd.notna(v) else 'NaN',
                'flops_M': lambda v: '{:.4f}'.format(v) if pd.notna(v) else 'NaN'}))
print()
print('saved (M units) -> main_results/table_46_literature_per_dataset.csv')
