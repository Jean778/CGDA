# -*- coding: utf-8 -*-
"""Measure computational overhead for Figure 4.8.1.

Outputs to REPO/main_results/:
  - panel_481a_pareto.csv      : performance vs latency per dataset
  - panel_481b_training_time.csv : training time distribution per method
  - panel_481c_calibration.csv  : calibration time vs label budget
  - panel_481d_latency.csv      : estimated latency across platforms
  - complexity_table.csv        : full complexity table
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import sys, time, math, statistics
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))

from revision_core import (SEEDS, StandardScaler, stratified_parts,
                           nested_calibration, check_disjoint,
                           train_network, network_output, Network)
from run_revision import load_drift, load_gsa, csv, shuffled_pool
from run_cdsl_full import load_cdsl

OUT = PROJECT / 'main_results'
OUT.mkdir(exist_ok=True)

BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
TARGET_METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup']


def count_macs(model, input_dim):
    """Estimate MACs for a linear network."""
    macs = 0
    # FE layers
    prev = input_dim
    for layer in model.fe:
        if isinstance(layer, nn.Linear):
            out = layer.out_features
            macs += prev * out
            prev = out
    # Classifier
    macs += prev * model.cls.out_features
    return macs


def measure_inference_latency(model, x, n_runs=200, warmup=20):
    """Measure single-sample inference latency in ms."""
    model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_tensor)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return statistics.mean(times), statistics.stdev(times)


def measure_calibration_time(dataset_name, budget, seed=42):
    """Measure time to adapt to new target domain (calibration + training)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    if dataset_name == 'UCI_Drift':
        x, y, b = load_drift(str(PROJECT))
        fold = 5
        target = np.flatnonzero(b == fold)
        source = np.flatnonzero(b < fold)
        classes = np.arange(1, 7)
        input_dim = x.shape[1]
    elif dataset_name == 'CDSL':
        x, y, source, target = load_cdsl('master_slave1')
        classes = np.arange(1, 7)
        input_dim = x.shape[1]
    else:
        raise ValueError(f'unknown dataset {dataset_name}')

    pool, test = stratified_parts(target, y, [.2, .8], seed)
    pool = shuffled_pool(pool, seed)
    cal = nested_calibration(pool, target, y, budget)

    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)

    steps = math.ceil((len(source) + len(pool) + 360) / 128)

    times = {}
    for method in TARGET_METHODS:
        t0 = time.perf_counter()
        model, details = train_network(xs, y[source], xc, y[cal], method, seed, classes, 50, steps)
        t1 = time.perf_counter()
        times[method] = (t1 - t0) * 1000  # ms

        # Measure inference latency
        lat_mean, lat_std = measure_inference_latency(model, xc[:1])
        times[f'{method}_inference_ms'] = lat_mean
        times[f'{method}_inference_std'] = lat_std

        # Parameter count
        times[f'{method}_params'] = sum(p.numel() for p in model.parameters())
        times[f'{method}_macs'] = count_macs(model, input_dim)

    times['budget'] = budget
    times['dataset'] = dataset_name
    times['seed'] = seed
    times['input_dim'] = input_dim
    times['n_classes'] = len(classes)
    return times


def main():
    print('=== measuring computational overhead ===')

    # 1. Measure calibration time across budgets
    cal_rows = []
    for dataset in ['UCI_Drift', 'CDSL']:
        print(f'  {dataset}...')
        for budget in BUDGETS:
            t = measure_calibration_time(dataset, budget, seed=42)
            cal_rows.append(t)
            print(f'    budget={budget}: done')

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT / 'panel_481c_calibration.csv', index=False)
    print(f'\npanel_481c: {len(cal_df)} rows')

    # 2. Aggregate training time from existing experiments
    print('\n=== aggregating training time from existing experiments ===')
    train_rows = []
    for dataset, task_dir in [('UCI_Drift', 'main_figure/drift_forward/tasks'),
                               ('CDSL', 'main_figure/cdsl_forward/tasks')]:
        root = PROJECT / 'results' / task_dir
        if not root.exists():
            continue
        for td in sorted(root.iterdir()):
            md = td / 'model_details.csv'
            if not md.exists():
                continue
            df = pd.read_csv(md)
            train_rows.append(df)

    train_df = pd.concat(train_rows, ignore_index=True)
    # Filter to b=0.1 for main comparison
    train_01 = train_df[train_df['budget'] == 0.1]
    train_summary = train_01.groupby(['dataset', 'method'])['seconds'].agg(
        ['mean', 'std', 'min', 'max', 'count']).reset_index()
    train_summary.columns = ['dataset', 'method', 'mean_s', 'std_s', 'min_s', 'max_s', 'n']
    train_summary.to_csv(OUT / 'panel_481b_training_time.csv', index=False)
    print(f'panel_481b: {len(train_summary)} rows')

    # 3. Pareto data: performance vs latency
    print('\n=== building pareto data ===')
    # Load panel a performance
    panel_a = pd.read_csv(OUT / 'panel_a_long.csv')
    a_01 = panel_a[panel_a['budget'] == 0.1] if 'budget' in panel_a.columns else panel_a

    # Merge with latency
    lat_summary = cal_df.groupby('dataset')[
        [f'{m}_inference_ms' for m in TARGET_METHODS]
    ].first().reset_index()

    # Reshape latency
    lat_long = []
    for _, row in lat_summary.iterrows():
        ds = row['dataset']
        for m in TARGET_METHODS:
            lat_long.append({
                'dataset': ds,
                'method': m,
                'inference_ms': row[f'{m}_inference_ms']
            })
    lat_df = pd.DataFrame(lat_long)

    # Merge with performance
    perf = a_01.groupby(['dataset', 'method'])['macro_f1'].mean().reset_index()
    pareto = perf.merge(lat_df, on=['dataset', 'method'])
    pareto.to_csv(OUT / 'panel_481a_pareto.csv', index=False)
    print(f'panel_481a: {len(pareto)} rows')

    # 4. Complexity table
    print('\n=== building complexity table ===')
    complexity_rows = []
    for _, row in cal_df.iterrows():
        if row['budget'] != 0.10:
            continue
        for m in TARGET_METHODS:
            params = row[f'{m}_params']
            macs = row[f'{m}_macs']
            weight_kb = params * 4 / 1024  # float32
            complexity_rows.append({
                'dataset': row['dataset'],
                'method': m,
                'input_dim': row['input_dim'],
                'n_classes': row['n_classes'],
                'params': params,
                'macs': macs,
                'weight_storage_kb_float32': round(weight_kb, 1),
                'weight_storage_kb_int8': round(params / 1024, 1),
                'inference_ms': round(row[f'{m}_inference_ms'], 4),
                'training_time_s': round(train_summary[
                    (train_summary['dataset'] == row['dataset']) &
                    (train_summary['method'] == m)
                ]['mean_s'].values[0], 3) if len(train_summary[
                    (train_summary['dataset'] == row['dataset']) &
                    (train_summary['method'] == m)
                ]) > 0 else None,
            })

    complexity_df = pd.DataFrame(complexity_rows)
    complexity_df.to_csv(OUT / 'complexity_table.csv', index=False)
    print(f'complexity_table: {len(complexity_df)} rows')

    # 5. Three-platform latency estimate (simulated)
    # Based on measured CPU latency, estimate relative performance
    platforms = pd.DataFrame([
        {'platform': 'CPU (this machine)', 'relative_speed': 1.0, 'note': 'measured'},
        {'platform': 'Edge CPU (estimated)', 'relative_speed': 0.3, 'note': 'estimated, 3x slower'},
        {'platform': 'MCU (estimated)', 'relative_speed': 0.05, 'note': 'estimated, 20x slower'},
    ])
    # Use UCI Drift cg_sup as reference
    ref_latency = cal_df[cal_df['dataset'] == 'UCI_Drift'][f'cg_sup_inference_ms'].values[0]
    platforms['estimated_latency_ms'] = platforms['relative_speed'] * ref_latency
    platforms.to_csv(OUT / 'panel_481d_latency.csv', index=False)
    print(f'panel_481d: {len(platforms)} rows')

    # ---------- sanity summary ----------
    print('\n=== sanity: complexity table ===')
    print(complexity_df[['dataset', 'method', 'params', 'macs', 'inference_ms', 'training_time_s']].to_string(index=False))

    print('\n=== sanity: pareto ===')
    print(pareto.to_string(index=False))

    print('\n=== sanity: calibration time by budget ===')
    print(cal_df[['dataset', 'budget', 'cg_sup', 'cg_sup_inference_ms']].to_string(index=False))


if __name__ == '__main__':
    main()
