"""Evaluation metrics and statistical aggregation for GasBench."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def ece(probs, y, classes, bins=10):
    """Expected calibration error with confidence exactly one in the final bin."""
    conf = probs.max(1)
    correct = np.asarray(classes)[probs.argmax(1)] == y
    ids = np.minimum((conf * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(ids == k) * abs(correct[ids == k].mean() - conf[ids == k].mean())
            for k in range(bins)
            if np.any(ids == k)
        )
    )


def classification_metrics(y, p, probs=None, classes=None):
    result = {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0),
    }
    if probs is not None:
        result["ece"] = ece(probs, y, classes)
    for c in np.unique(y):
        result[f"f1_class_{c}"] = f1_score(y, p, labels=[c], average=None, zero_division=0)[0]
    return result


def oscr(y, p, d, known):
    """Open-set classification rate: CCR against unknown false acceptance.

    Ties in distance are grouped; low distance means "known". Returns
    (area_under_curve, curve_dataframe).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    d = np.asarray(d)
    known = np.asarray(known, bool)
    assert known.any() and (~known).any()
    order = np.argsort(d, kind="stable")
    ds = d[order]
    ends = np.r_[np.flatnonzero(np.diff(ds) != 0), len(ds) - 1]
    correct = (known & (y == p))[order]
    ccr = np.r_[0, np.cumsum(correct)[ends] / known.sum()]
    fpr = np.r_[0, np.cumsum((~known)[order])[ends] / (~known).sum()]
    thr = np.r_[np.nextafter(ds[0], -np.inf), ds[ends]]
    curve = pd.DataFrame(
        {"threshold": thr, "unknown_false_acceptance": fpr, "correct_classification_rate": ccr}
    )
    return float(np.trapezoid(ccr, fpr)), curve


def rejection_metrics(y, p, d, known, threshold):
    a = d <= threshold
    k = np.asarray(known, bool)
    return {
        "known_acceptance": float(a[k].mean()),
        "unknown_rejection": float((~a[~k]).mean()),
        "known_correct_acceptance": float((a[k] & (p[k] == y[k])).mean()),
        "unknown_false_acceptance": float(a[~k].mean()),
    }


def binary_metrics(y, d, threshold):
    pred = (d > threshold).astype(int)
    return {
        "auroc": roc_auc_score(y, d),
        "average_precision": average_precision_score(y, d),
        "event_f1": f1_score(y, pred, zero_division=0),
        "false_alarm_rate": pred[y == 0].mean(),
        "event_sensitivity": pred[y == 1].mean(),
        "event_miss_rate": 1 - pred[y == 1].mean(),
        "accuracy": accuracy_score(y, pred),
    }


def quantile_threshold(d, q):
    assert len(d) > 0 and np.isfinite(d).all()
    return float(np.quantile(d, q, method="higher"))


def seed_summary(metrics, keys):
    """Mean over folds within each seed, then mean and sample SD across seeds.

    Folds are not treated as independent replicates (no pseudo-replication).
    Returns (per_seed_frame, aggregated_frame).
    """
    g = metrics.groupby(keys + ["seed", "metric"], dropna=False)["value"]
    per = g.agg(value="mean", n_folds="count").reset_index()
    agg = (
        per.groupby(keys + ["metric"], dropna=False)["value"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1), n_seeds="count")
        .reset_index()
    )
    return per, agg
