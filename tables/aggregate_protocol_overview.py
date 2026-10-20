# -*- coding: utf-8 -*-
"""Aggregate Figure 1 (Sec 4.2) panel data.

Outputs to REPO/main_results/:
  - panel_a_long.csv     : A/B/C datasets x methods x seed x fold, macro_f1 (panel a)
  - panel_b_delta.csv    : cg_sup - cal_sup per (dataset, fold, seed) (panel b)
  - panel_c_budget.csv   : A/B datasets x budget x method x seed x fold (panel c)
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
OUT = ROOT / 'main_results'
OUT.mkdir(exist_ok=True)

TARGET_METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup']
GSA_METHODS = ['base', 'cal_sup', 'cg_sup']  # no mixup on GSA


def load_main_figure(dataset_dir, dataset_name):
    """Load all metrics.csv from results/main_figure/<dataset>/tasks/*/metrics.csv."""
    rows = []
    task_root = ROOT / 'results/main_figure' / dataset_dir / 'tasks'
    for task_dir in sorted(task_root.iterdir()):
        mf = task_dir / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out['dataset'] = dataset_name
    return out


def load_gsa():
    """Load GSA MixedCV from revision_v2."""
    rows = []
    task_root = ROOT / 'results/revision_v2/tasks'
    for task_dir in sorted(task_root.iterdir()):
        if not task_dir.name.startswith('gsa_cv_stratified_cv_b2_'):
            continue
        mf = task_dir / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out['dataset'] = 'C_GSA'
    return out


def main():
    # 1. Load all
    drift = load_main_figure('drift_forward', 'A_UCI_Drift')
    cdsl = load_main_figure('cdsl_forward', 'B_CDSL')
    gsa = load_gsa()
    print(f'loaded: drift={len(drift)} rows, cdsl={len(cdsl)} rows, gsa={len(gsa)} rows')

    all_df = pd.concat([drift, cdsl, gsa], ignore_index=True)
    mf = all_df[all_df['metric'] == 'macro_f1'].copy()
    mf = mf[mf['threshold_rule'] == 'none']
    print(f'macro_f1 rows: {len(mf)}')

    # ---------- panel a: fixed budget (b=0.1 for A/B, b=2 for C) ----------
    a_drift = mf[(mf['dataset'] == 'A_UCI_Drift') & (mf['budget'] == 0.10)]
    a_cdsl = mf[(mf['dataset'] == 'B_CDSL') & (mf['budget'] == 0.10)]
    a_gsa = mf[mf['dataset'] == 'C_GSA']
    panel_a = pd.concat([a_drift, a_cdsl, a_gsa], ignore_index=True)
    panel_a_out = panel_a[['dataset', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    panel_a_out = panel_a_out.sort_values(['dataset', 'method', 'seed', 'fold']).reset_index(drop=True)
    panel_a_out.to_csv(OUT / 'panel_a_long.csv', index=False)
    print(f'panel_a: {len(panel_a_out)} rows -> {OUT / "panel_a_long.csv"}')

    # ---------- panel b: cg_sup - cal_sup per (dataset, fold, seed) ----------
    wide = panel_a.pivot_table(
        index=['dataset', 'seed', 'fold'],
        columns='method', values='value'
    ).reset_index()
    wide['delta_cg_sup_minus_cal_sup'] = wide['cg_sup'] - wide['cal_sup']
    panel_b = wide[['dataset', 'seed', 'fold', 'delta_cg_sup_minus_cal_sup']].sort_values(
        ['dataset', 'fold', 'seed']).reset_index(drop=True)
    panel_b.to_csv(OUT / 'panel_b_delta.csv', index=False)
    print(f'panel_b: {len(panel_b)} rows -> {OUT / "panel_b_delta.csv"}')

    # ---------- panel c: budget curves for A and B only ----------
    panel_c = mf[mf['dataset'].isin(['A_UCI_Drift', 'B_CDSL'])].copy()
    panel_c_out = panel_c[['dataset', 'budget', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    panel_c_out = panel_c_out.sort_values(['dataset', 'budget', 'method', 'seed', 'fold']).reset_index(drop=True)
    panel_c_out.to_csv(OUT / 'panel_c_budget.csv', index=False)
    print(f'panel_c: {len(panel_c_out)} rows -> {OUT / "panel_c_budget.csv"}')

    # ---------- quick sanity summary ----------
    print('\n=== sanity: mean macro_f1 by dataset/method (panel a, b=0.1 / b=2) ===')
    summary = panel_a.groupby(['dataset', 'method'])['value'].agg(['mean', 'std', 'count']).round(4)
    print(summary.to_string())

    print('\n=== sanity: panel b mean delta ===')
    print(panel_b.groupby('dataset')['delta_cg_sup_minus_cal_sup'].agg(['mean', 'std', 'count']).round(4).to_string())

    print('\n=== sanity: panel c by dataset/budget/method (mean) ===')
    s = panel_c_out.groupby(['dataset', 'budget', 'method'])['macro_f1'].mean().round(4).unstack()
    print(s.to_string())


if __name__ == '__main__':
    main()
