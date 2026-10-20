# -*- coding: utf-8 -*-
"""Aggregate all cosine OSR results into same format as previous panel_44_*.csv.

Outputs:
  - panel_44a_oscr_cosine.csv
  - panel_44b_contamination_cosine.csv
  - panel_44c_distribution_cosine.csv
  - panel_44_sample_scores_cosine.csv
  - panel_44_thresholds_cosine.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = REPO
OUT = ROOT / 'main_results'

METHODS = ['base', 'cal_sup', 'cg_sup']
BUDGETS = [0.01, 0.05, 0.1, 0.15, 0.2]


def calc_oscr_curve(distance, is_unknown):
    """Compute OSCR curve: unknown_fpr vs known_ccr."""
    order = np.argsort(distance)
    ds = distance[order]
    known = is_unknown[order]

    ends = np.r_[np.flatnonzero(np.diff(ds) != 0), len(ds) - 1]
    ccr = np.r_[0, np.cumsum(known == 0)[ends] / np.sum(known == 0)]
    fpr = np.r_[0, np.cumsum(known == 1)[ends] / np.sum(known == 1)]
    return fpr, ccr


def main():
    oscr_rows = []
    contamination_rows = []
    distribution_rows = []
    sample_rows = []
    threshold_rows = []
    metric_rows = []

    for method in METHODS:
        for budget in BUDGETS:
            task_root = ROOT / 'results/osr_cosine' / method / f'b{budget}' / 'tasks'
            if not task_root.exists():
                print(f'WARNING: {task_root} not found')
                continue

            for td in sorted(task_root.iterdir()):
                if not td.is_dir():
                    continue
                task_id = td.name
                parts = task_id.split('_')
                unknown_pair = parts[1]
                seed = int(parts[3][1:])
                fold = int(parts[4][1:])

                # predictions.csv
                pred_file = td / 'predictions.csv'
                if not pred_file.exists():
                    continue
                pred = pd.read_csv(pred_file)

                # 1. per-sample scores
                pred['protocol'] = 'OSR'
                pred['direction'] = unknown_pair
                pred['budget'] = budget
                pred['seed'] = seed
                pred['fold'] = fold
                pred['method'] = method
                pred['unknown_pair'] = unknown_pair
                pred['y_pred'] = pred['y_true']  # placeholder

                sample_rows.append(pred[['protocol', 'direction', 'budget', 'seed', 'fold',
                                         'method', 'unknown_pair', 'sample_id', 'is_unknown',
                                         'distance_score', 'y_true', 'y_pred']])

                # 2. OSCR curve
                fpr, ccr = calc_oscr_curve(pred.distance_score.values, pred.is_unknown.values)
                for i in range(len(fpr)):
                    oscr_rows.append({
                        'setting': unknown_pair,
                        'method': method,
                        'seed': seed,
                        'fold': fold,
                        'unknown_fpr': fpr[i],
                        'known_ccr': ccr[i],
                    })

                # 3. distance distribution (representative: seed=42, fold=5)
                if seed == 42 and fold == 5:
                    for _, row in pred.iterrows():
                        distribution_rows.append({
                            'setting': unknown_pair,
                            'method': method,
                            'seed': seed,
                            'fold': fold,
                            'distance': row.distance_score,
                            'is_known': 1 - row.is_unknown,
                        })

                # 4. contamination ratio + threshold
                known_scores = pred[pred.is_unknown == 0].distance_score.values
                unknown_scores = pred[pred.is_unknown == 1].distance_score.values

                contamination_ratios = [0.0, 0.25, 0.50, 0.75]

                for rho in contamination_ratios:
                    n_known = len(known_scores)
                    n_unknown = int(n_known * rho / (1 - rho)) if rho < 1 else len(unknown_scores)
                    n_unknown = min(n_unknown, len(unknown_scores))

                    if n_unknown > 0:
                        mixed_pool = np.concatenate([known_scores, unknown_scores[:n_unknown]])
                    else:
                        mixed_pool = known_scores.copy()

                    thr_q90 = np.percentile(mixed_pool, 90)

                    known_accept = np.mean(known_scores <= thr_q90)
                    unknown_reject = np.mean(unknown_scores > thr_q90)
                    unknown_false_accept = np.mean(unknown_scores <= thr_q90)

                    contamination_rows.append({
                        'setting': unknown_pair,
                        'method': method,
                        'strategy': 'full',
                        'rho': rho,
                        'seed': seed,
                        'fold': fold,
                        'known_correct_acceptance': known_accept,
                        'unknown_rejection': unknown_reject,
                        'unknown_false_acceptance': unknown_false_accept,
                    })

                    threshold_rows.append({
                        'protocol': 'OSR',
                        'direction': unknown_pair,
                        'budget': budget,
                        'seed': seed,
                        'fold': fold,
                        'method': method,
                        'unknown_pair': unknown_pair,
                        'contamination_ratio': rho,
                        'threshold_method': f'mixed{int(rho*100):02d}_full_q90',
                        'threshold': thr_q90,
                        'n_reference': len(mixed_pool),
                    })

                # pure-known reference
                thr_known = np.percentile(known_scores, 90)
                known_accept_known = np.mean(known_scores <= thr_known)
                unknown_reject_known = np.mean(unknown_scores > thr_known)
                unknown_false_accept_known = np.mean(unknown_scores <= thr_known)

                contamination_rows.append({
                    'setting': unknown_pair,
                    'method': method,
                    'strategy': 'known_reference',
                    'rho': 0.0,
                    'seed': seed,
                    'fold': fold,
                    'known_correct_acceptance': known_accept_known,
                    'unknown_rejection': unknown_reject_known,
                    'unknown_false_acceptance': unknown_false_accept_known,
                })

                threshold_rows.append({
                    'protocol': 'OSR',
                    'direction': unknown_pair,
                    'budget': budget,
                    'seed': seed,
                    'fold': fold,
                    'method': method,
                    'unknown_pair': unknown_pair,
                    'contamination_ratio': 0.0,
                    'threshold_method': 'known_reference_q90',
                    'threshold': thr_known,
                    'n_reference': len(known_scores),
                })

                # 5. metric summary
                n1 = np.sum(pred.is_unknown == 1)
                n0 = np.sum(pred.is_unknown == 0)
                ranks = np.argsort(np.argsort(pred.distance_score.values)) + 1
                u1 = np.sum(ranks[pred.is_unknown == 1]) - n1 * (n1 + 1) / 2
                auroc = u1 / (n1 * n0) if (n1 * n0) > 0 else np.nan

                metric_rows.append({
                    'method': method,
                    'budget': budget,
                    'setting': unknown_pair,
                    'seed': seed,
                    'fold': fold,
                    'auroc': auroc,
                })

    # save
    print('Saving...')

    sample_df = pd.concat(sample_rows, ignore_index=True)
    sample_df.to_csv(OUT / 'panel_44_sample_scores_cosine.csv', index=False)
    print(f'panel_44_sample_scores_cosine.csv: {len(sample_df)} rows')

    oscr_df = pd.DataFrame(oscr_rows)
    oscr_df.to_csv(OUT / 'panel_44a_oscr_cosine.csv', index=False)
    print(f'panel_44a_oscr_cosine.csv: {len(oscr_df)} rows')

    dist_df = pd.DataFrame(distribution_rows)
    dist_df.to_csv(OUT / 'panel_44c_distribution_cosine.csv', index=False)
    print(f'panel_44c_distribution_cosine.csv: {len(dist_df)} rows')

    contamination_df = pd.DataFrame(contamination_rows)
    contamination_df.to_csv(OUT / 'panel_44b_contamination_cosine.csv', index=False)
    print(f'panel_44b_contamination_cosine.csv: {len(contamination_df)} rows')

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUT / 'panel_44_thresholds_cosine.csv', index=False)
    print(f'panel_44_thresholds_cosine.csv: {len(threshold_df)} rows')

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT / 'panel_44_metrics_cosine.csv', index=False)
    print(f'panel_44_metrics_cosine.csv: {len(metric_df)} rows')

    # print summary
    print()
    print('=== AUROC summary ===')
    for method in METHODS:
        sub = metric_df[metric_df.method == method]
        print(f'{method}: {sub.auroc.mean():.4f} ± {sub.auroc.std():.4f}')

    print()
    print('done')


if __name__ == '__main__':
    main()
