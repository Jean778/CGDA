# -*- coding: utf-8 -*-
"""Aggregate Figure 4.8.2 (Sec 4.8.2) panel data.

Outputs to REPO/main_results/:
  - panel_482a_delta_performance.csv : paired delta macro_f1 vs fp32
  - panel_482b_delta_ece.csv        : paired delta ECE vs fp32
  - panel_482c_params_vs_perf.csv   : param retention vs perf change
  - panel_482d_config_summary.csv    : per-config summary
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
OUT = ROOT / 'main_results'
QUANT_ROOT = ROOT / 'results/quant_v1'


def main():
    # Load all folds data
    af = pd.read_csv(QUANT_ROOT / 'all_folds.csv')
    print(f'loaded {len(af)} rows from all_folds.csv')
    print(f'configs: {sorted(af.config.unique())}')

    # ---------- panel a: paired delta macro_f1 ----------
    print('\n=== panel a: paired delta macro_f1 ===')
    # Pivot: each (seed, fold) has all configs
    pivot = af.pivot_table(index=['seed', 'fold'], columns='config', values='macro_f1').reset_index()

    base = 'fp32'
    configs = [c for c in sorted(af.config.unique()) if c != base]

    rows = []
    for config in configs:
        if config not in pivot.columns or base not in pivot.columns:
            continue
        tmp = pivot[['seed', 'fold', base, config]].copy()
        tmp['delta_macro_f1'] = tmp[config] - tmp[base]
        tmp['config'] = config
        tmp = tmp.rename(columns={base: 'fp32_macro_f1', config: 'config_macro_f1'})
        rows.append(tmp[['seed', 'fold', 'config', 'fp32_macro_f1', 'config_macro_f1', 'delta_macro_f1']])

    panel_a = pd.concat(rows, ignore_index=True)
    panel_a = panel_a.sort_values(['config', 'seed', 'fold']).reset_index(drop=True)
    panel_a.to_csv(OUT / 'panel_482a_delta_performance.csv', index=False)
    print(f'panel_482a: {len(panel_a)} rows')

    # ---------- panel b: paired delta ECE ----------
    print('\n=== panel b: paired delta ECE ===')
    pivot_ece = af.pivot_table(index=['seed', 'fold'], columns='config', values='ece').reset_index()

    rows = []
    for config in configs:
        if config not in pivot_ece.columns or base not in pivot_ece.columns:
            continue
        tmp = pivot_ece[['seed', 'fold', base, config]].copy()
        tmp['delta_ece'] = tmp[config] - tmp[base]
        tmp['config'] = config
        tmp = tmp.rename(columns={base: 'fp32_ece', config: 'config_ece'})
        rows.append(tmp[['seed', 'fold', 'config', 'fp32_ece', 'config_ece', 'delta_ece']])

    panel_b = pd.concat(rows, ignore_index=True)
    panel_b = panel_b.sort_values(['config', 'seed', 'fold']).reset_index(drop=True)
    panel_b.to_csv(OUT / 'panel_482b_delta_ece.csv', index=False)
    print(f'panel_482b: {len(panel_b)} rows')

    # ---------- panel c: params vs perf ----------
    print('\n=== panel c: params vs perf ===')
    # Compute mean delta per config
    param_count = af.groupby('config')['parameter_count'].first()
    base_params = param_count['fp32']

    summary = panel_a.groupby('config')['delta_macro_f1'].agg(
        ['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
    summary['param_count'] = summary['config'].map(param_count)
    summary['param_retention'] = summary['param_count'] / base_params
    summary['param_reduction_pct'] = (1 - summary['param_retention']) * 100

    # Also ECE change
    ece_summary = panel_b.groupby('config')['delta_ece'].mean().reset_index()
    ece_summary.columns = ['config', 'mean_delta_ece']
    summary = summary.merge(ece_summary, on='config')

    summary.to_csv(OUT / 'panel_482c_params_vs_perf.csv', index=False)
    print(f'panel_482c: {len(summary)} rows')

    # ---------- panel d: config summary (all metrics) ----------
    print('\n=== panel d: config summary ===')
    config_summary = af.groupby('config').agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_macro_f1=('macro_f1', 'mean'),
        std_macro_f1=('macro_f1', 'std'),
        mean_ece=('ece', 'mean'),
        std_ece=('ece', 'std'),
        param_count=('parameter_count', 'first'),
        bits=('bits', 'first'),
        mean_time_s=('seconds', 'mean'),
    ).reset_index()
    config_summary['param_retention'] = config_summary['param_count'] / base_params
    config_summary['param_reduction_pct'] = (1 - config_summary['param_retention']) * 100
    config_summary.to_csv(OUT / 'panel_482d_config_summary.csv', index=False)
    print(f'panel_482d: {len(config_summary)} rows')

    # ---------- sanity summary ----------
    print('\n=== sanity: delta macro_f1 by config ===')
    print(panel_a.groupby('config')['delta_macro_f1'].agg(
        ['mean', 'std', 'median', 'min', 'max']).round(4).to_string())

    print('\n=== sanity: delta ECE by config ===')
    print(panel_b.groupby('config')['delta_ece'].agg(
        ['mean', 'std', 'median', 'min', 'max']).round(4).to_string())

    print('\n=== sanity: params vs perf ===')
    print(summary[['config', 'param_count', 'param_reduction_pct', 'mean', 'std']].round(4).to_string())


if __name__ == '__main__':
    main()
