# -*- coding: utf-8 -*-
"""Measure GSA complexity + build both pareto variants (latency & training time).

Adds GSA rows to complexity_table.csv, and generates:
  - panel_481a_pareto_latency.csv   : macro_f1 vs inference_ms (A/B/C)
  - panel_481a_pareto_training.csv  : macro_f1 vs training_time_s (A/B/C)
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

from revision_core import (StandardScaler, check_disjoint,
                           train_network, network_output, Network)
from run_revision import load_gsa

OUT = PROJECT / 'main_results'
TARGET_METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup']


def count_macs(model, input_dim):
    macs = 0
    prev = input_dim
    for layer in model.fe:
        if isinstance(layer, nn.Linear):
            out = layer.out_features
            macs += prev * out
            prev = out
    macs += prev * model.cls.out_features
    return macs


def measure_inference_latency(model, x, n_runs=200, warmup=20):
    model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_tensor)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times)


def main():
    # ---- 1. GSA complexity measurement (train + latency) ----
    print('=== measuring GSA overhead ===', flush=True)
    x, y, conc = load_gsa(str(PROJECT))
    classes = np.arange(6)
    input_dim = x.shape[1]

    target_conc = 50
    source = np.flatnonzero(conc != target_conc)
    target = np.flatnonzero(conc == target_conc)

    seed = 42
    rng = np.random.RandomState(seed)
    cal_ids = []
    for c in classes:
        ids = target[y[target] == c]
        n = min(2, len(ids) - 1)
        cal_ids.extend(rng.choice(ids, size=n, replace=False).tolist())
    cal = np.array(cal_ids)
    test = np.setdiff1d(target, cal)
    check_disjoint(source_train=source, calibration=cal, test=test)

    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)

    steps = math.ceil((len(source) + len(cal) + 360) / 128)

    gsa_rows = []
    for m in TARGET_METHODS:
        t0 = time.perf_counter()
        model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, 50, steps)
        train_s = time.perf_counter() - t0
        lat_mean, lat_std = measure_inference_latency(model, xc[:1])
        params = sum(p.numel() for p in model.parameters())
        macs = count_macs(model, input_dim)
        gsa_rows.append({
            'dataset': 'GSA',
            'method': m,
            'input_dim': input_dim,
            'n_classes': len(classes),
            'params': params,
            'macs': macs,
            'weight_storage_kb_float32': round(params * 4 / 1024, 1),
            'weight_storage_kb_int8': round(params / 1024, 1),
            'inference_ms': round(lat_mean, 4),
            'training_time_s': round(train_s, 3),
            'seed': seed,
        })
        print(f'  {m}: params={params} macs={macs} lat={lat_mean:.4f}ms train={train_s:.3f}s', flush=True)

    gsa_df = pd.DataFrame(gsa_rows)

    # ---- 2. Extend complexity_table with GSA rows ----
    comp_path = OUT / 'complexity_table.csv'
    comp = pd.read_csv(comp_path)
    # drop previous GSA rows (re-run safe), then append fresh GSA
    comp = comp[comp['dataset'] != 'GSA']
    cols = ['dataset', 'method', 'input_dim', 'n_classes', 'params', 'macs',
            'weight_storage_kb_float32', 'weight_storage_kb_int8', 'inference_ms', 'training_time_s']
    gsa_small = gsa_df[cols]
    comp_new = pd.concat([comp, gsa_small], ignore_index=True)
    comp_new.to_csv(comp_path, index=False)
    print(f'\ncomplexity_table: {len(comp_new)} rows (was {len(comp)})', flush=True)

    # ---- 3. Build both pareto variants ----
    # Performance: panel_correct_protocols_summary has A/B; panel_gsa_loco_summary has C.
    print('\n=== building pareto variants ===', flush=True)

    # A/B performance (correct protocols summary, budget=0.1)
    prot = pd.read_csv(OUT / 'panel_correct_protocols_summary.csv')
    print('prot cols:', prot.columns.tolist(), flush=True)

    # C performance (gsa loco summary, budget=0.02)
    gsa_sum = pd.read_csv(OUT / 'panel_gsa_loco_summary.csv')
    print('gsa_sum cols:', gsa_sum.columns.tolist(), flush=True)

    # Latency: A/B from panel_481c_calibration (budget 0.10), C from gsa_df
    cal = pd.read_csv(OUT / 'panel_481c_calibration.csv')
    cal_10 = cal[cal['budget'] == 0.10]
    lat_rows = []
    for _, row in cal_10.iterrows():
        ds = 'A_UCI_Drift' if row['dataset'] == 'UCI_Drift' else 'B_CDSL'
        for m in TARGET_METHODS:
            lat_rows.append({'dataset': ds, 'method': m,
                             'inference_ms': row[f'{m}_inference_ms']})
    for _, row in gsa_df.iterrows():
        lat_rows.append({'dataset': 'C_GSA', 'method': row['method'],
                         'inference_ms': row['inference_ms']})
    lat_df = pd.DataFrame(lat_rows)

    # Training time: A/B from panel_481b_training_time (budget 0.1 mean),
    #                C from panel_gsa_loco_long seconds (avg over target_conc)
    tt = pd.read_csv(OUT / 'panel_481b_training_time.csv')
    gl = pd.read_csv(OUT / 'panel_gsa_loco_long.csv')
    gsa_train = gl.groupby('method')['seconds'].mean().to_dict()
    tt_rows = []
    for _, row in tt.iterrows():
        ds = 'A_UCI_Drift' if row['dataset'] == 'UCI_Drift' else 'B_CDSL'
        tt_rows.append({'dataset': ds, 'method': row['method'],
                        'training_time_s': row['mean_s']})
    for m in TARGET_METHODS:
        tt_rows.append({'dataset': 'C_GSA', 'method': m,
                        'training_time_s': gsa_train.get(m, np.nan)})
    tt_df = pd.DataFrame(tt_rows)

    # Performance:
    # A/B: panel_a_long (A_UCI_Drift / B_CDSL), avg macro_f1
    # C: GSA LOCO summary, avg over target_conc
    pa = pd.read_csv(OUT / 'panel_a_long.csv')
    a_perf = pa[pa['dataset'] == 'A_UCI_Drift'].groupby('method')['macro_f1'].mean().to_dict()
    b_perf = pa[pa['dataset'] == 'B_CDSL'].groupby('method')['macro_f1'].mean().to_dict()
    perf_rows = []
    for m in TARGET_METHODS:
        perf_rows.append({'dataset': 'A_UCI_Drift', 'method': m,
                          'macro_f1': a_perf.get(m, np.nan)})
        perf_rows.append({'dataset': 'B_CDSL', 'method': m,
                          'macro_f1': b_perf.get(m, np.nan)})
    gsa_sum = pd.read_csv(OUT / 'panel_gsa_loco_summary.csv')
    for m, g in gsa_sum.groupby('method'):
        if m not in TARGET_METHODS:
            continue
        perf_rows.append({'dataset': 'C_GSA', 'method': m,
                          'macro_f1': g['macro_f1_mean'].mean()})
    perf_df = pd.DataFrame(perf_rows)

    # Merge
    pareto_lat = perf_df.merge(lat_df, on=['dataset', 'method'])
    pareto_lat = pareto_lat.merge(tt_df, on=['dataset', 'method'], how='left')
    pareto_lat = pareto_lat.sort_values(['dataset', 'method']).reset_index(drop=True)
    pareto_lat.to_csv(OUT / 'panel_481a_pareto_latency.csv', index=False)
    print('panel_481a_pareto_latency:', len(pareto_lat), 'rows', flush=True)

    pareto_train = perf_df.merge(tt_df, on=['dataset', 'method'])
    pareto_train = pareto_train.sort_values(['dataset', 'method']).reset_index(drop=True)
    pareto_train.to_csv(OUT / 'panel_481a_pareto_training.csv', index=False)
    print('panel_481a_pareto_training:', len(pareto_train), 'rows', flush=True)

    print('\n=== pareto_latency ===', flush=True)
    print(pareto_lat.to_string(index=False), flush=True)
    print('\n=== pareto_training ===', flush=True)
    print(pareto_train.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
