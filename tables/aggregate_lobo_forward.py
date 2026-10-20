# -*- coding: utf-8 -*-
"""Aggregate LOBO vs Forward summary for Sec 4.2.

Outputs to REPO/main_results/:
  - panel_summary_lobo_forward.csv : combined long table
  - panel_summary_lobo_forward_mean.csv : mean per protocol/budget/method
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd

ROOT = REPO
OUT = ROOT / 'main_results'


def load_tasks(root_dir, protocol_name):
    rows = []
    task_root = ROOT / 'results' / root_dir / 'tasks'
    if not task_root.exists():
        return pd.DataFrame()
    for td in sorted(task_root.iterdir()):
        mf = td / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        df['protocol'] = protocol_name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    # Load Forward (main_figure)
    fwd = load_tasks('main_figure/drift_forward', 'Forward')
    print(f'Forward: {len(fwd)} rows')

    # Load LOBO (main_figure/drift_lobo)
    lobo = load_tasks('main_figure/drift_lobo', 'LOBO')
    print(f'LOBO: {len(lobo)} rows')

    # Combine
    all_df = pd.concat([fwd, lobo], ignore_index=True)

    # Filter to macro_f1
    mf = all_df[all_df['metric'] == 'macro_f1'].copy()
    mf = mf[mf['threshold_rule'] == 'none']

    # Output long table
    long = mf[['protocol', 'budget', 'method', 'seed', 'fold', 'value']].rename(
        columns={'value': 'macro_f1'})
    long = long.sort_values(['protocol', 'budget', 'method', 'seed', 'fold']).reset_index(drop=True)
    long.to_csv(OUT / 'panel_summary_lobo_forward.csv', index=False)
    print(f'long: {len(long)} rows -> {OUT / "panel_summary_lobo_forward.csv"}')

    # Output mean summary
    mean_summary = long.groupby(['protocol', 'budget', 'method'])['macro_f1'].agg(
        ['mean', 'std', 'count']).reset_index()
    mean_summary = mean_summary.sort_values(['protocol', 'budget', 'method']).reset_index(drop=True)
    mean_summary.to_csv(OUT / 'panel_summary_lobo_forward_mean.csv', index=False)
    print(f'mean: {len(mean_summary)} rows -> {OUT / "panel_summary_lobo_forward_mean.csv"}')

    # ---------- sanity ----------
    print('\n=== sanity: mean macro_f1 by protocol/budget/method ===')
    pivot = mean_summary.pivot_table(index=['protocol', 'budget'], columns='method', values='mean')
    print(pivot.round(4).to_string())


if __name__ == '__main__':
    main()
