"""Experiment 3 -- INT8 quantization / pruning coupling with CGDA calibration
training, under the closed LOBO protocol of revision_v2.

Protocol (identical to run_revision.run_closed):
  source = 9 batches, target = 1 batch (leave-one-batch-out)
  pool,test = stratified_parts(target, [.2,.8]); cal = nested_calibration(pool, budget)
  StandardScaler fit on source; cg_sup = centroid Mixup + calibration supervision
  metrics: accuracy, macro-F1, ECE  (5 seeds x 10 folds per configuration)

Configurations:
  fp32        cg_sup FP32 (authoritative baseline, re-run here for pairing)
  ptq         FP32 -> INT8 fake-quant, no retrain
  ptq_ft      FP32 -> INT8 + 5-epoch source-domain fine-tune
  qat         INT8 fake-quant training from scratch
  prune30_ptq FP32 -> prune 30% -> INT8, no retrain
  prune30_qat FP32 -> prune 30% -> INT8 + full fine-tune (deployment recalib)

Output: results/quant_v1/summary_mean_sd.csv + per-fold csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os, sys, math, time, json
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
sys.path.insert(0, str(PROJECT / "benchmark"))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from revision_core import (SEEDS, stratified_parts, nested_calibration,
                           check_disjoint, train_network, network_output,
                           classification_metrics, seed_summary)
from run_revision import load_drift
from gasbench.quant import train_quant

CONFIGS = ["fp32", "w8_ptq", "w8_ft", "w8_qat",
           "p30_w8_ptq", "p30_w8_ft", "w8a8_ptq"]
BUDGET = 0.1          # calibration budget used by the authoritative cg_sup runs
FOLDS = list(range(1, 11))


def run_fold(x, y, b, fold, seed, out):
    target = np.flatnonzero(b == fold)
    source = np.flatnonzero(b != fold)
    pool, test = stratified_parts(target, y, [.2, .8], seed)
    pool = pool.copy()
    np.random.RandomState(seed + 888).shuffle(pool)
    cal = nested_calibration(pool, target, y, BUDGET)
    check_disjoint(source=source, calibration=cal, test=test)

    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    classes = np.arange(1, 7)
    # Calibration-derived per-feature input scale for the INT8-input contrast
    # (99th quantile of |pool| per channel; long tails make per-tensor max
    # scales collapse, which is exactly what the contrast demonstrates).
    xp = sc.transform(x[pool]).astype(np.float32)
    in_scale_feat = np.maximum(np.quantile(np.abs(xp), 0.99, axis=0), 1e-6)

    fp32_model = None
    for cfg in CONFIGS:
        t0 = time.time()
        if cfg == "fp32":
            model, details = train_network(xs, y[source], xc, y[cal], "cg_sup",
                                           seed, classes, 50, steps)
            fp32_model = model
        elif cfg == "w8_ptq":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, ft_epochs=0,
                                         pretrained=fp32_model)
        elif cfg == "w8_ft":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, ft_epochs=5,
                                         pretrained=fp32_model)
        elif cfg == "w8_qat":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, epochs=50, steps=steps)
        elif cfg == "p30_w8_ptq":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, prune=0.3, ft_epochs=0,
                                         pretrained=fp32_model)
        elif cfg == "p30_w8_ft":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, prune=0.3, ft_epochs=50,
                                         pretrained=fp32_model)
        elif cfg == "w8a8_ptq":
            model, details = train_quant(xs, y[source], xc, y[cal], seed, classes,
                                         quantize=True, ft_epochs=0,
                                         pretrained=fp32_model,
                                         in_scale=in_scale_feat)
        p, pr, _ = network_output(model, xt, classes)
        m = classification_metrics(y[test], p, pr, classes)
        m["seconds"] = time.time() - t0
        m["config"] = cfg
        m["fold"] = fold
        m["seed"] = seed
        m.update({k: v for k, v in details.items()
                  if k in ("parameter_count", "epochs", "optimizer_steps", "bits")})
        out.append(m)
        print(f"  fold={fold} seed={seed} {cfg:10s} acc={m['accuracy']:.4f} "
              f"mf1={m['macro_f1']:.4f} ece={m['ece']:.4f} "
              f"params={details['parameter_count']} {m['seconds']:.1f}s", flush=True)


def main():
    out_dir = PROJECT / "results" / "quant_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, b = load_drift(PROJECT)
    rows = []
    n_tasks = len(CONFIGS) * len(FOLDS) * len(SEEDS)
    t0_all = time.time()
    done = 0
    for seed in SEEDS:
        for fold in FOLDS:
            done += 1
            print(f"\n=== seed={seed} fold={fold} ({done}/{len(SEEDS)*len(FOLDS)}) "
                  f"elapsed {time.time()-t0_all:.0f}s ===", flush=True)
            run_fold(x, y, b, fold, seed, rows)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_folds.csv", index=False, float_format="%.6g")

    # aggregate: equal fold mean per seed, then SD across seeds (revision_v2 style)
    long = df.melt(id_vars=["config", "seed", "fold"],
                   value_vars=["accuracy", "macro_f1", "ece", "seconds",
                               "parameter_count"],
                   var_name="metric")
    per, agg = seed_summary(long, ["config"])
    per.to_csv(out_dir / "per_seed.csv", index=False, float_format="%.6g")
    agg.to_csv(out_dir / "summary_mean_sd.csv", index=False, float_format="%.6g")

    print("\n=== SUMMARY (mean±sd across 5 seeds) ===")
    piv = agg.pivot_table(index="config", columns="metric", values="mean")
    sd = agg.pivot_table(index="config", columns="metric", values="std")
    for cfg in CONFIGS:
        if cfg in piv.index:
            print(f"{cfg:12s} acc={piv.loc[cfg,'accuracy']:.4f}±{sd.loc[cfg,'accuracy']:.4f} "
                  f"mf1={piv.loc[cfg,'macro_f1']:.4f} ece={piv.loc[cfg,'ece']:.4f} "
                  f"params={piv.loc[cfg,'parameter_count']:.0f}")
    print(f"\ntotal {n_tasks} tasks in {time.time()-t0_all:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
