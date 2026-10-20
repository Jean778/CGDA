# -*- coding: utf-8 -*-
"""Aggregate Figure 4.3 (Sec 4.3) panel data.

Outputs to REPO/main_results/:
  - panel_43a_ablation.csv  : ablation comparison (cal_sup / mixup_std / cg_mix / cg_sup)
  - panel_43b_centroid.csv  : correct vs shuffled centroid (cg_sup / cg_shuffle)
  - panel_43c_features.csv  : intra-class distance & inter-class separation (base vs cg_sup)
  - panel_43d_baselines.csv : mean_shift vs base (gain heatmap)
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
OUT = ROOT / 'main_results'


def load_from(dataset_dir, dataset_name):
    """Load all metrics.csv from results/main_figure_v2/<dataset>/tasks/*/metrics.csv."""
    rows = []
    task_root = ROOT / 'results/main_figure_v2' / dataset_dir / 'tasks'
    if not task_root.exists():
        return pd.DataFrame()
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


def load_features(dataset_dir, dataset_name):
    """Load feature_metrics.csv from all tasks."""
    rows = []
    task_root = ROOT / 'results/main_figure_v2' / dataset_dir / 'tasks'
    if not task_root.exists():
        return pd.DataFrame()
    for task_dir in sorted(task_root.iterdir()):
        fm = task_dir / 'feature_metrics.csv'
        if not fm.exists():
            continue
        df = pd.read_csv(fm)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out['dataset'] = dataset_name
    return out


def main():
    # Load v2 (new methods)
    drift_v2 = load_from('drift_forward', 'A_UCI_Drift')
    cdsl_v2 = load_from('cdsl_forward', 'B_CDSL')
    v2 = pd.concat([drift_v2, cdsl_v2], ignore_index=True)

    # Load v1 (existing methods: base, cal_sup, mixup_std, cg_sup)
    drift_v1_rows = []
    cdsl_v1_rows = []
    for d, lst in [('drift_forward', drift_v1_rows), ('cdsl_forward', cdsl_v1_rows)]:
        task_root = ROOT / 'results/main_figure' / d / 'tasks'
        if task_root.exists():
            for task_dir in sorted(task_root.iterdir()):
                mf = task_dir / 'metrics.csv'
                if mf.exists():
                    lst.append(pd.read_csv(mf))
    drift_v1 = pd.concat(drift_v1_rows, ignore_index=True) if drift_v1_rows else pd.DataFrame()
    cdsl_v1 = pd.concat(cdsl_v1_rows, ignore_index=True) if cdsl_v1_rows else pd.DataFrame()
    drift_v1['dataset'] = 'A_UCI_Drift'
    cdsl_v1['dataset'] = 'B_CDSL'
    v1 = pd.concat([drift_v1, cdsl_v1], ignore_index=True)

    # Merge v1 and v2
    all_df = pd.concat([v1, v2], ignore_index=True)
    mf = all_df[all_df['metric'] == 'macro_f1'].copy()
    mf = mf[mf['threshold_rule'] == 'none']
    print(f'total macro_f1 rows: {len(mf)}')
    print(f'methods: {sorted(mf.method.unique())}')

    # ---------- panel a: ablation at b=0.1 ----------
    ablation_methods = ['cal_sup', 'mixup_std', 'cg_mix', 'cg_sup']
    panel_a = mf[(mf['budget'] == 0.10) & (mf['method'].isin(ablation_methods))].copy()
    panel_a_out = panel_a[['dataset', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    panel_a_out = panel_a_out.sort_values(['dataset', 'method', 'seed', 'fold']).reset_index(drop=True)
    panel_a_out.to_csv(OUT / 'panel_43a_ablation.csv', index=False)
    print(f'\npanel_43a: {len(panel_a_out)} rows')

    # ---------- panel b: correct vs shuffled centroid ----------
    centroid_methods = ['cg_sup', 'cg_shuffle']
    panel_b = mf[(mf['budget'] == 0.10) & (mf['method'].isin(centroid_methods))].copy()
    panel_b_out = panel_b[['dataset', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    panel_b_out = panel_b_out.sort_values(['dataset', 'method', 'seed', 'fold']).reset_index(drop=True)
    panel_b_out.to_csv(OUT / 'panel_43b_centroid.csv', index=False)
    print(f'panel_43b: {len(panel_b_out)} rows')

    # ---------- panel c: feature space metrics ----------
    drift_f = load_features('drift_forward', 'A_UCI_Drift')
    cdsl_f = load_features('cdsl_forward', 'B_CDSL')
    feat = pd.concat([drift_f, cdsl_f], ignore_index=True)
    if len(feat) > 0:
        feat_out = feat[['dataset', 'method', 'seed', 'fold', 'metric', 'value']].rename(
            columns={'value': 'metric_value'})
        feat_out = feat_out.sort_values(['dataset', 'method', 'metric', 'seed', 'fold']).reset_index(drop=True)
        feat_out.to_csv(OUT / 'panel_43c_features.csv', index=False)
        print(f'panel_43c: {len(feat_out)} rows')
    else:
        print('panel_43c: no feature metrics found')

    # ---------- panel d: mean_shift vs base ----------
    baseline_methods = ['base', 'mean_shift']
    panel_d = mf[mf['method'].isin(baseline_methods)].copy()
    panel_d_out = panel_d[['dataset', 'budget', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    panel_d_out = panel_d_out.sort_values(['dataset', 'budget', 'method', 'seed', 'fold']).reset_index(drop=True)
    panel_d_out.to_csv(OUT / 'panel_43d_baselines.csv', index=False)
    print(f'panel_43d: {len(panel_d_out)} rows')

    # ---------- sanity summary ----------
    print('\n=== sanity: panel a ablation (mean macro_f1 at b=0.1) ===')
    print(panel_a_out.groupby(['dataset', 'method'])['macro_f1'].mean().round(4).unstack().to_string())

    print('\n=== sanity: panel b centroid comparison (mean) ===')
    print(panel_b_out.groupby(['dataset', 'method'])['macro_f1'].mean().round(4).unstack().to_string())

    if len(feat) > 0:
        print('\n=== sanity: panel c feature metrics (mean) ===')
        print(feat_out.groupby(['dataset', 'method', 'metric'])['metric_value'].mean().round(4).unstack().to_string())

    print('\n=== sanity: panel d baselines (mean by budget) ===')
    print(panel_d_out.groupby(['dataset', 'budget', 'method'])['macro_f1'].mean().round(4).unstack().to_string())


if __name__ == '__main__':
    main()
