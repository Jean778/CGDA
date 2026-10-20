
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

# -*- coding: utf-8 -*-
import pandas as pd, os
OUT = str(REPO / 'main_results')

# === Table 4.5a: AP by direction (5-seed mean) ===
a = pd.read_csv(os.path.join(OUT, 'table_45a_ap_summary_cosine.csv'))
methods = [('mlp', 'MLP\n(no cal)'),
           ('threshold_only', 'Threshold-only\n(target Q95)'),
           ('cg_train_threshold', 'CGDA + target thr'),
           ('cg_full', 'CGDA (full)')]
rows = []
for _, r in a.iterrows():
    row = {'方向': r.direction, 'n_test': int(r.n_test),
           'n_event': int(r.n_event), '事件比例': f'{r.event_ratio*100:.1f}%'}
    for m, lab in methods:
        k = m + '_mean'
        s = m + '_std'
        row[lab] = f'{r[k]:.3f} ({r[s]:.3f})' if k in a.columns else '—'
    rows.append(row)
tab_a = pd.DataFrame(rows)
tab_a.to_csv(os.path.join(OUT, 'table_45a_paper.csv'), index=False, encoding='utf-8-sig')
print('=== Table 4.5a ===')
print(tab_a.to_string(index=False))
print()

# === Table 4.5b: Delta AP ===
b = pd.read_csv(os.path.join(OUT, 'panel_45b_delta_ap_cosine.csv'))
focus = ['threshold_only', 'shift_only', 'shift_threshold', 'cg_full']
labmap = {'threshold_only': 'Threshold-only (no shift)',
          'shift_only': 'Shift-only (source thr)',
          'shift_threshold': 'Shift + target thr',
          'cg_full': 'CGDA (full)'}
rows = []
for m in focus:
    s = b[b.method == m]['delta_AP']
    rows.append({'方法': labmap[m], 'ΔAP mean': round(s.mean(), 4),
                 'ΔAP std': round(s.std(), 4)})
tab_b = pd.DataFrame(rows)
tab_b.to_csv(os.path.join(OUT, 'table_45b_paper.csv'), index=False, encoding='utf-8-sig')
print('=== Table 4.5b ===')
print(tab_b.to_string(index=False))
print()

# === Table 4.5c: fixed-threshold decision cost ===
c = pd.read_csv(os.path.join(OUT, 'panel_45c_pr_points_cosine.csv'))
order = [('mlp', 'MLP (source-val Q95)'),
         ('threshold_only', 'Threshold-only (target Q95)'),
         ('shift_only', 'Shift-only (source Q95)'),
         ('shift_threshold', 'Shift + target Q95'),
         ('cg_train', 'CG-train (source Q95)'),
         ('cg_train_threshold', 'CG + target Q95'),
         ('cg_full', 'CGDA (full: shift + shifted target Q95)')]
rows = []
for m, lab in order:
    s = c[c.method == m]
    rows.append({'方法 / 阈值策略': lab,
                 '事件检出率 (↑)': f'{s.event_sensitivity.mean():.3f} ± {s.event_sensitivity.std():.3f}',
                 '误报率 (↓)': f'{s.false_alarm_rate.mean():.3f} ± {s.false_alarm_rate.std():.3f}'})
tab_c = pd.DataFrame(rows)
tab_c.to_csv(os.path.join(OUT, 'table_45c_paper.csv'), index=False, encoding='utf-8-sig')
print('=== Table 4.5c ===')
print(tab_c.to_string(index=False))
