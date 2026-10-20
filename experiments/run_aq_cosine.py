# -*- coding: utf-8 -*-
"""AQ event detection with cosine distance instead of Mahalanobis.

Same structure as run_aq in run_revision.py: same methods, same scores dict,
same threshold rules, same output files. Only the distance function changes:
  distances(f, params)  [Mahalanobis to class centroid]
  ->
  cosine_dist_to_centroid(f, centroid)

Outputs to REPO/results/aq_cosine/tasks/<setting>_s<seed>/
  16 directions x 5 seeds = 80 tasks
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import sys, time, math, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))

from revision_core import (StandardScaler, PCA, check_disjoint, train_network,
                           network_output, quantile_threshold, binary_metrics)
from run_revision import load_aq, chronological_domain_blocks, Recorder, csv

SEEDS = [7, 42, 123, 2024, 2025]
DIRECTIONS = [(A, B) for A in range(4) for B in range(4)]
EPOCHS = 80  # same as run_aq (aq uses 80 when epochs==50)


def cosine_dist_to_centroid(feats, centroid):
    """1 - cosine similarity between each feature and the normal-class centroid."""
    feats = np.asarray(feats, float)
    centroid = np.asarray(centroid, float)
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    cn = centroid / (np.linalg.norm(centroid) + 1e-8)
    return 1.0 - fn @ cn


def run_task(task):
    A, B = task['setting']
    seed = task['seed']
    out_dir = PROJECT / 'results/aq_cosine/tasks' / f'aq_{A}-{B}_s{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / 'completion.csv').exists():
        return task['task_id'], 'cached', 0

    t0 = time.time()
    try:
        x, c, b, rawids, times, tr, sv, cn, ct, te, broad, normal, event, low, high = \
            __import__('run_revision', fromlist=['aq_setup']).aq_setup(str(PROJECT), A, B, seed)

        sc = StandardScaler().fit(x[tr])
        xs = sc.transform(x[tr]).astype(np.float32)
        xv = sc.transform(x[sv]).astype(np.float32)
        xc = sc.transform(x[cn]).astype(np.float32)
        xct = sc.transform(x[ct]).astype(np.float32)
        xb = sc.transform(x[broad]).astype(np.float32)
        yt = event[tr].astype(int)
        in_strict = np.isin(broad, te)

        r = Recorder(task, out_dir)
        r.split(source_train=rawids[tr], source_threshold=rawids[sv],
                target_train_calibration=rawids[cn], target_threshold_calibration=rawids[ct],
                test=rawids[broad])
        csv({'feature': np.arange(x.shape[1]), 'mean': sc.mean_, 'scale': sc.scale_},
            out_dir / 'preprocessing.csv')
        csv({'reference': ['CO', 'C6H6', 'NOx', 'NO2'], 'normal_p50': low, 'event_p95': high},
            out_dir / 'label_thresholds.csv')

        scores = {}
        thresholds = []

        pca = PCA(n_components=.95).fit(xs[yt == 0])
        recon = lambda z: np.square(z - pca.inverse_transform(pca.transform(z))).sum(1)
        scores['pca'] = (recon(xb), quantile_threshold(recon(xv), .95))

        steps = math.ceil((len(tr) + 30 + 40) / 128)

        for mode in ['base', 'cal_sup', 'cg_sup']:
            model, details = train_network(xs, yt, xc, np.zeros(len(cn), int), mode, seed,
                                           [0, 1], epochs=EPOCHS, steps=steps, small=True)
            _, _, fs = network_output(model, xs, [0, 1])
            _, _, fv = network_output(model, xv, [0, 1])
            _, _, fc = network_output(model, xc, [0, 1])
            _, _, fct = network_output(model, xct, [0, 1])
            _, _, fb = network_output(model, xb, [0, 1])

            # normal-class centroid from source normal features
            mu_s = fs[yt == 0].mean(0)
            # shift: target calibration normal mean - source normal mean
            shift = fc.mean(0) - mu_s

            # distances (cosine): larger = more abnormal
            ds = cosine_dist_to_centroid(fb, mu_s)
            dct = cosine_dist_to_centroid(fct, mu_s)
            dsv = cosine_dist_to_centroid(fv, mu_s)
            shifted = cosine_dist_to_centroid(fb - shift, mu_s)
            shiftedct = cosine_dist_to_centroid(fct - shift, mu_s)

            src_thr = quantile_threshold(dsv, .95)
            tgt_thr = quantile_threshold(dct, .95)
            shift_thr = quantile_threshold(shiftedct, .95)

            if mode == 'base':
                scores.update({'mlp': (ds, src_thr),
                               'threshold_only': (ds, tgt_thr),
                               'shift_only': (shifted, src_thr),
                               'shift_threshold': (shifted, shift_thr)})
            elif mode == 'cal_sup':
                scores.update({'cal_sup': (ds, src_thr),
                               'cal_sup_threshold': (ds, tgt_thr)})
            else:
                scores.update({'cg_train': (ds, src_thr),
                               'cg_train_threshold': (ds, tgt_thr),
                               'cg_full': (shifted, shift_thr)})

            csv({'sample_id': rawids[ct], 'distance_unshifted': dct,
                 'distance_shifted': shiftedct}, out_dir / f'threshold_scores_{mode}.csv')
            csv({'sample_id': rawids[sv], 'distance': dsv},
                out_dir / f'source_validation_scores_{mode}.csv')
            csv({'feature': np.arange(len(shift)), 'shift': shift},
                out_dir / f'feature_shift_{mode}.csv')
            r.configrow(mode, n_source=len(tr), n_source_normal=int(np.sum(yt == 0)),
                        n_calibration_training=len(cn), n_calibration_threshold=len(ct),
                        n_strict_test=len(te), n_all_test=len(broad),
                        source_domain=A, target_domain=B, **details)

        for method, (d, thr) in scores.items():
            for name, mask in [('extreme', in_strict),
                               ('including_intermediate', np.ones(len(broad), bool))]:
                y = event[broad][mask].astype(int)
                r.metric(method, binary_metrics(y, d[mask], thr), evaluation=name,
                         n_test=int(mask.sum()))
            r.predictions(method, rawids[broad], event[broad].astype(int), (d > thr).astype(int),
                          distance=d, in_extreme_task=in_strict.astype(int),
                          timestamp=times[broad], threshold=thr)
            thresholds.append({'method': method, 'threshold': thr})
        csv(thresholds, out_dir / 'thresholds.csv')
        r.finish()

        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0

    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(
            out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def main():
    tasks = [dict(setting=(A, B), seed=s, task_id=f'aq_{A}-{B}_s{s}')
             for A, B in DIRECTIONS for s in SEEDS]
    print(f'PLAN tasks={len(tasks)} (16 dirs x 5 seeds)', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete':
                done += 1
            elif st == 'failed':
                fail += 1
            print(f'[{i}/{len(tasks)}] {st} {tid} {el:.1f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
