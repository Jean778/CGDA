# -*- coding: utf-8 -*-
"""Aggregate correct-protocol results (6 methods) for Sec 4.2.

Merges:
  - drift_lobo_correct / drift_fwd_single  (base, cal_sup, mixup_std, cg_sup)
  - drift_lobo_correct_extra / drift_fwd_single_extra  (cg_mix, cg_shuffle)

Outputs to REPO/main_results/:
  - panel_correct_protocols_long.csv
  - panel_correct_protocols_summary.csv
  - sample_counts.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
OUT = ROOT / 'main_results'


def load_tasks(root_dir, protocol):
    rows = []
    task_root = ROOT / 'results/main_figure' / root_dir / 'tasks'
    if not task_root.exists():
        return pd.DataFrame()
    for td in sorted(task_root.iterdir()):
        mf = td / 'metrics.csv'
        if not mf.exists():
            continue
        df = pd.read_csv(mf)
        df['protocol'] = protocol
        parts = td.name.split('_')
        budget = float(parts[1][1:])
        seed = int(parts[2][1:])
        direction = parts[3]  # '1to2' or '5to6'
        src, tgt = direction.split('to')
        df['source_batch'] = int(src)
        df['target_batch'] = int(tgt)
        df['direction'] = direction
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    # Load base (4 methods) + extra (2 methods) for both protocols
    lobo_base = load_tasks('drift_lobo_correct', 'LOBO')
    lobo_extra = load_tasks('drift_lobo_correct_extra', 'LOBO')
    fwd_base = load_tasks('drift_fwd_single', 'Forward')
    fwd_extra = load_tasks('drift_fwd_single_extra', 'Forward')

    all_df = pd.concat([lobo_base, lobo_extra, fwd_base, fwd_extra], ignore_index=True)
    print(f'total rows: {len(all_df)}')

    # Filter macro_f1
    mf = all_df[all_df['metric'] == 'macro_f1'].copy()
    mf = mf[mf['threshold_rule'] == 'none']
    print(f'macro_f1 rows: {len(mf)}')
    print(f'methods: {sorted(mf.method.unique())}')

    # Pivot
    pivot = mf.pivot_table(
        index=['protocol', 'source_batch', 'target_batch', 'direction', 'budget', 'seed'],
        columns='method', values='value'
    ).reset_index()

    # Compute gains
    pivot['cg_minus_cal'] = pivot['cg_sup'] - pivot['cal_sup']
    pivot['cg_minus_base'] = pivot['cg_sup'] - pivot['base']

    # Save long table
    long_cols = ['protocol', 'source_batch', 'target_batch', 'direction', 'budget', 'seed',
                 'base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle',
                 'cg_minus_cal', 'cg_minus_base']
    long_out = pivot[long_cols].sort_values(['protocol', 'direction', 'budget', 'seed']).reset_index(drop=True)
    long_out.to_csv(OUT / 'panel_correct_protocols_long.csv', index=False)
    print(f'long: {len(long_out)} rows')

    # Summary
    summary = long_out.groupby(['protocol', 'source_batch', 'target_batch', 'direction', 'budget']).agg(
        base_mean=('base', 'mean'),
        cal_sup_mean=('cal_sup', 'mean'),
        mixup_std_mean=('mixup_std', 'mean'),
        cg_sup_mean=('cg_sup', 'mean'),
        cg_mix_mean=('cg_mix', 'mean'),
        cg_shuffle_mean=('cg_shuffle', 'mean'),
        cg_minus_cal_mean=('cg_minus_cal', 'mean'),
        n_seeds=('seed', 'count')
    ).reset_index()
    summary = summary.sort_values(['protocol', 'direction', 'budget']).reset_index(drop=True)
    summary.to_csv(OUT / 'panel_correct_protocols_summary.csv', index=False)
    print(f'summary: {len(summary)} rows')

    # Sample counts
    sample_rows = []
    for protocol, root_dir in [('LOBO', 'drift_lobo_correct'), ('Forward', 'drift_fwd_single')]:
        task_root = ROOT / 'results/main_figure' / root_dir / 'tasks'
        seen_directions = set()
        for td in sorted(task_root.iterdir()):
            md = td / 'model_details.csv'
            if not md.exists():
                continue
            df = pd.read_csv(md)
            direction = td.name.split('_')[3]
            if direction in seen_directions:
                continue
            seen_directions.add(direction)
            row = df.iloc[0]
            sample_rows.append({
                'protocol': protocol,
                'direction': direction,
                'source_batch': int(direction.split('to')[0]),
                'target_batch': int(direction.split('to')[1]),
                'n_source': row['n_source'],
                'n_test': row['n_test'],
                'n_cal_b0.1': row['n_target_labeled'],
            })
    sample_df = pd.DataFrame(sample_rows).sort_values(['protocol', 'direction']).reset_index(drop=True)
    sample_df.to_csv(OUT / 'sample_counts.csv', index=False)
    print(f'sample_counts: {len(sample_df)} rows')

    # ---------- sanity ----------
    print('\n=== sanity: mean cg_sup by protocol/budget ===')
    s = summary.groupby(['protocol', 'budget'])['cg_sup_mean'].mean().round(4)
    print(s.to_string())

    print('\n=== sanity: all methods at b=0.1 ===')
    b01 = summary[summary.budget == 0.10]
    print(b01.groupby('protocol')[['base_mean', 'cal_sup_mean', 'mixup_std_mean',
                                   'cg_sup_mean', 'cg_mix_mean', 'cg_shuffle_mean']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
