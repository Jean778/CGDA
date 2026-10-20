# -*- coding: utf-8 -*-
"""Build final deployment comparison table (lightweight literature models + CGDA).

Sources:
  lit metrics : table_46_lit_metrics_light.csv   (macro_f1, accuracy, ece, params_M, inference_ms)
  lit peakmem : table_46_lit_peakmem.csv         (peak_rss_MB)
  ours fixed  : table_46_cgda_fixed_test.csv     (CGDA same fixed test set: f1/acc/ece/inference_ms/params_M/peak_rss_MB)

Output: main_results/table_46_deployment_light.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd, os

OUT = str(REPO / 'main_results')

lit = pd.read_csv(os.path.join(OUT, 'table_46_lit_metrics_light.csv'))
lit = lit.drop_duplicates(subset=['dataset', 'model', 'seed', 'direction'], keep='last')
pm = pd.read_csv(os.path.join(OUT, 'table_46_lit_peakmem.csv'))
pm = pm[['dataset', 'model', 'peak_rss_MB']]

m = lit.merge(pm, on=['dataset', 'model'], how='left')

rows = []
for _, r in m.iterrows():
    rows.append({
        'dataset': r['dataset'], 'model': r['model'], 'method': 'base',
        'macro_f1': r['macro_f1'], 'accuracy': r['accuracy'], 'ece': r['ece'],
        'params_M': r['params_M'], 'peak_rss_MB': r['peak_rss_MB'],
        'inference_ms': r['inference_ms'], 'direction': r['direction'],
    })

# CGDA from fixed-test measurement (same test sets as lit models)
cg = pd.read_csv(os.path.join(OUT, 'table_46_cgda_fixed_test.csv'))
for _, r in cg.iterrows():
    rows.append({
        'dataset': r['dataset'], 'model': 'CGDA (ours)', 'method': 'cg_sup',
        'macro_f1': r['macro_f1'], 'accuracy': r['accuracy'], 'ece': r['ece'],
        'params_M': r['params_M'], 'peak_rss_MB': r['peak_rss_MB'],
        'inference_ms': r['inference_ms'],
        'direction': ('1to2' if r['dataset'] in ('A', 'B') else '50ppb'),
        'budget': r['budget'],
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, 'table_46_deployment_light.csv'), index=False, float_format='%.4f')
print('saved -> table_46_deployment_light.csv | rows:', len(out))
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 30)
print(out[['dataset', 'model', 'method', 'macro_f1', 'accuracy', 'ece',
           'params_M', 'peak_rss_MB', 'inference_ms', 'direction', 'budget']].to_string(index=False))
