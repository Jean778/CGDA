# -*- coding: utf-8 -*-
"""Build formal tables for 4.5 (cosine): AP summary + fixed-threshold decision table."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

OUT = REPO / 'main_results'

# ---------- AP summary table (6 progressive dirs, 5-seed mean +/- std) ----------
a = pd.read_csv(OUT / 'panel_45a_ap_heatmap_cosine.csv')
b = pd.read_csv(OUT / 'panel_45b_delta_ap_cosine.csv')
ts = pd.read_csv(OUT / 'panel_45d_timeseries_cosine.csv')

# per-seed AP (need seed-level; recompute from per-task metrics)
root = REPO / 'results' / 'aq_cosine' / 'tasks'
seeds = [7, 42, 123, 2024, 2025]
dirs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
KEY = ['mlp', 'threshold_only', 'cal_sup', 'cal_sup_threshold', 'cg_full', 'cg_train_threshold']

rows = []
for A, B in dirs:
    for s in seeds:
        mf = root / f'aq_{A}-{B}_s{s}' / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        df = df[(df['evaluation'] == 'extreme') & (df['metric'] == 'average_precision')]
        for _, r in df.iterrows():
            rows.append({'A': A, 'B': B, 'seed': s, 'method': r['method'], 'AP': r['value']})
m = pd.DataFrame(rows)
m['direction'] = m['A'].astype(str) + '->' + m['B'].astype(str)

# event ratio per direction (seed 7 timeseries)
prop = ts.groupby(['A', 'B']).apply(
    lambda g: pd.Series({'n_test': len(g), 'n_event': int(g.y_true.sum()),
                         'event_ratio': g.y_true.mean()}), include_groups=False).reset_index()
prop['direction'] = prop['A'].astype(str) + '->' + prop['B'].astype(str)

pivot = m[m['method'].isin(KEY)].pivot_table(index='direction', columns='method',
                                             values='AP', aggfunc=['mean', 'std'])
tab = pd.DataFrame(index=KEY)
tab['direction'] = ''
# build long table instead: direction x method mean+std
recs = []
for (A, B), g in m[m['method'].isin(KEY)].groupby(['A', 'B']):
    d = f"{A}->{B}"
    prow = prop[(prop['A'] == A) & (prop['B'] == B)]
    ev = prow['event_ratio'].values[0]
    nt = int(prow['n_test'].values[0])
    ne = int(prow['n_event'].values[0])
    recs.append({'direction': d, 'n_test': nt, 'n_event': ne, 'event_ratio': round(ev, 4)})
    for meth in KEY:
        sub = g[g['method'] == meth]['AP']
        recs[-1][f'{meth}_mean'] = round(sub.mean(), 4)
        recs[-1][f'{meth}_std'] = round(sub.std(), 4)
t1 = pd.DataFrame(recs)
t1.to_csv(OUT / 'table_45a_ap_summary_cosine.csv', index=False)
print('=== Table 45a: AP summary (5-seed mean +/- std) ===')
print(t1.to_string(index=False))

# ---------- Fixed-threshold decision table (0->1 focus + all dirs) ----------
pr = pd.read_csv(OUT / 'panel_45c_pr_points_cosine.csv')
pr['direction'] = pr['A'].astype(str) + '->' + pr['B'].astype(str)
d01 = pr[pr['direction'] == '0->1']
DEC = ['mlp', 'threshold_only', 'cal_sup', 'cg_full']
recs2 = []
for meth in DEC:
    sub = d01[d01['method'] == meth]
    sens = sub['event_sensitivity'].mean()
    far = sub['false_alarm_rate'].mean()
    # F1 at fixed threshold per seed, then mean
    f1s = []
    for s in seeds:
        ss = sub[sub['seed'] == s]
        yp = ss['event_sensitivity'].values[0]  # not enough; need counts
    recs2.append({'method': meth,
                  'event_sensitivity': round(sens, 4),
                  'false_alarm_rate': round(far, 4)})
t2 = pd.DataFrame(recs2)
t2.to_csv(OUT / 'table_45c_threshold_decision_cosine.csv', index=False)
print()
print('=== Table 45c: 0->1 fixed-threshold decision (5-seed mean) ===')
print(t2.to_string(index=False))

# per-direction summary of best-fixed-threshold methods
print()
print('=== all dirs fixed-threshold (cg_full) ===')
cg = pr[pr['method'] == 'cg_full'].groupby('direction')[['event_sensitivity', 'false_alarm_rate']].mean().round(4)
print(cg.to_string())
