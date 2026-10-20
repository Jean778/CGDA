# -*- coding: utf-8 -*-
"""Aggregate single-source multi-target experiments into wide format.

Output:
  - panel_single_source_long.csv
  - panel_single_source_wide.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
TASK_ROOT = ROOT / 'results/single_source'
OUT = ROOT / 'main_results'

METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle', 'cg_minus_cal', 'cg_minus_base']


def main():
    rows = []

    for method_dir in TASK_ROOT.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name

        for budget_dir in method_dir.iterdir():
            if not budget_dir.is_dir():
                continue
            budget = float(budget_dir.name[1:])  # b0.01 -> 0.01

            for task_dir in budget_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                metrics_file = task_dir / 'metrics.csv'
                if not metrics_file.exists():
                    continue

                metrics = pd.read_csv(metrics_file)
                rows.append(metrics)

    # long table
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.sort_values(['source_batch', 'target_batch', 'budget', 'seed']).reset_index(drop=True)
    long_df.to_csv(OUT / 'panel_single_source_long.csv', index=False)
    print(f'panel_single_source_long.csv: {len(long_df)} rows')

    # wide table (one column per method)
    wide_df = long_df.pivot_table(
        index=['source_batch', 'target_batch', 'direction', 'budget', 'seed'],
        columns='method',
        values='macro_f1'
    ).reset_index()

    # ensure column order
    cols = ['source_batch', 'target_batch', 'direction', 'budget', 'seed'] + METHODS
    for c in METHODS:
        if c not in wide_df.columns:
            wide_df[c] = np.nan
    wide_df = wide_df[cols]

    wide_df.to_csv(OUT / 'panel_single_source_wide.csv', index=False)
    print(f'panel_single_source_wide.csv: {len(wide_df)} rows')

    print()
    print('=== summary (macro F1) ===')
    summary = wide_df.groupby('budget')[METHODS].mean()
    print(summary)


if __name__ == '__main__':
    main()
