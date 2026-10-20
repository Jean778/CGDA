# -*- coding: utf-8 -*-
"""Aggregate cosine AQ results for 6 progressive directions into main_results."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
import numpy as np
from pathlib import Path

root = REPO / 'results' / 'aq_cosine' / 'tasks'
OUT = REPO / 'main_results'
dirs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
seeds = [7, 42, 123, 2024, 2025]

# ---- panel_45a: AP per direction x method (mean over 5 seeds) ----
met_rows = []
for A, B in dirs:
    for s in seeds:
        mf = root / f'aq_{A}-{B}_s{s}' / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        df = df[(df['evaluation'] == 'extreme') & (df['metric'] == 'average_precision')]
        for _, r in df.iterrows():
            met_rows.append({'A': A, 'B': B, 'seed': s, 'method': r['method'], 'ap': r['value']})
met = pd.DataFrame(met_rows)

pivot = met.pivot_table(index=['A', 'B'], columns='method', values='ap', aggfunc='mean').reset_index()
pivot['direction'] = pivot.apply(lambda r: f"{int(r['A'])}->{int(r['B'])}", axis=1)
pivot['is_same_domain'] = pivot['A'] == pivot['B']
order = ['A', 'B', 'direction', 'is_same_domain'] + [c for c in pivot.columns if c not in ['A', 'B', 'direction', 'is_same_domain']]
pivot = pivot[order]
pivot.to_csv(OUT / 'panel_45a_ap_heatmap_cosine.csv', index=False)
print('panel_45a_cosine rows:', len(pivot))

# ---- panel_45b: delta AP vs mlp (source-only base) ----
deltas = []
for A, B in dirs:
    for s in seeds:
        sub = met[(met['A'] == A) & (met['B'] == B) & (met['seed'] == s)]
        base_ap = sub[sub['method'] == 'mlp']['ap'].values
        if len(base_ap) == 0:
            continue
        base_ap = base_ap[0]
        for _, r in sub.iterrows():
            if r['method'] == 'mlp':
                continue
            deltas.append({'A': A, 'B': B, 'seed': s, 'method': r['method'],
                           'AP_base': base_ap, 'AP_method': r['ap'],
                           'delta_AP': r['ap'] - base_ap})
d = pd.DataFrame(deltas)
d.to_csv(OUT / 'panel_45b_delta_ap_cosine.csv', index=False)
print('panel_45b_cosine rows:', len(d))

# ---- panel_45c: event sensitivity / false alarm rate ----
pr_rows = []
for A, B in dirs:
    for s in seeds:
        mf = root / f'aq_{A}-{B}_s{s}' / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        df = df[df['evaluation'] == 'extreme']
        ev = df[df['metric'] == 'event_sensitivity'][['method', 'value']].set_index('method')['value']
        fa = df[df['metric'] == 'false_alarm_rate'][['method', 'value']].set_index('method')['value']
        for m in ev.index:
            pr_rows.append({'A': A, 'B': B, 'seed': s, 'method': m,
                            'event_sensitivity': ev[m], 'false_alarm_rate': fa[m]})
pr = pd.DataFrame(pr_rows)
pr.to_csv(OUT / 'panel_45c_pr_points_cosine.csv', index=False)
print('panel_45c_cosine rows:', len(pr))

# ---- panel_45d: timeseries (seed 7, cg_full) ----
ts_rows = []
for A, B in dirs:
    pf = root / f'aq_{A}-{B}_s7' / 'predictions_cg_full.csv'
    if not pf.exists():
        continue
    df = pd.read_csv(pf)
    df['A'] = A
    df['B'] = B
    df['method'] = 'cg_full'
    ts_rows.append(df)
ts = pd.concat(ts_rows, ignore_index=True)
ts.to_csv(OUT / 'panel_45d_timeseries_cosine.csv', index=False)
print('panel_45d_cosine rows:', len(ts))

print()
print('=== progressive directions AP (5-seed mean) ===')
print(pivot.round(4).to_string(index=False))
