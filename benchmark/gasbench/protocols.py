"""Evaluation protocols for GasBench.

Six protocol families are provided, mirroring the paper's evaluation:

* ``closed``     — leave-one-batch-out drift classification on UCI Drift
* ``budget``     — same as closed with calibration budgets 5% / 20%
* ``forward``    — past-to-future chronological drift on UCI Drift
* ``singlesource`` — one source batch adapted to every other batch
* ``osr``        — open-set rejection (known vs unknown classes) on UCI Drift
* ``gsa``        — cross-concentration transfer on GSA-LC (+ ``gsa_cv`` 5-fold)
* ``aq``         — temperature/humidity domain transfer and event detection

Every protocol writes the same CSV contract (metrics.csv, model_details.csv,
splits.csv, preprocessing.csv, ...), so downstream aggregation is uniform.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from .utils import SEEDS, check_disjoint, stratified_parts, nested_calibration
from .evaluate import (
    classification_metrics,
    oscr,
    rejection_metrics,
    binary_metrics,
    quantile_threshold,
    seed_summary,
)
from .methods import (
    train_network,
    network_output,
    distance_model,
    distances,
    list_methods,
)

PROTOCOL_VERSION = "gasbench_v1_20260915"
MAIN_METHODS = ["base", "cal_sup", "cg_mix", "cg_sup", "mixup_std", "cg_shuffle",
                "dann", "logreg", "sklearn_mlp", "mean_shift"]
SINGLE_SOURCE_METHODS = ["base", "cal_sup", "cg_sup", "cg_shuffle", "dann"]
KEYS = ["family", "setting", "budget", "method", "threshold_rule", "evaluation"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(df)
    temp = path.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False, float_format="%.12g")
    temp.replace(path)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class Recorder:
    """Collect metrics/config rows and flush them as CSV files."""

    def __init__(self, task, out):
        self.t = task
        self.out = out
        self.metrics = []
        self.config = []

    def metric(self, method, values, threshold_rule="none", evaluation="heldout", **extras):
        for k, v in values.items():
            self.metrics.append({**self.t, "method": method, "threshold_rule": threshold_rule,
                                 "evaluation": evaluation, "metric": k, "value": float(v), **extras})

    def configrow(self, method, **kwargs):
        self.config.append({**self.t, "method": method, **kwargs})

    def finish(self):
        csv(self.metrics, self.out / "metrics.csv")
        csv(self.config, self.out / "model_details.csv")

    def predictions(self, method, ids, y, p, pr=None, **cols):
        d = {"sample_id": ids, "y_true": y, "y_pred": p, **cols}
        if pr is not None:
            for i in range(pr.shape[1]):
                d[f"p_class_position_{i}"] = pr[:, i]
        csv(d, self.out / f"predictions_{method}.csv")

    def split(self, **parts):
        rows = [{"role": k, "sample_id": int(i)} for k, z in parts.items() for i in z]
        csv(rows, self.out / "splits.csv")


def shuffled_pool(pool, seed):
    z = pool.copy()
    np.random.RandomState(seed + 888).shuffle(z)
    return z


# ---------------------------------------------------------------------------
# Protocol runners
# ---------------------------------------------------------------------------
def _neural_or_lr(xs, ys, xc, yc, xt, method, seed, classes, epochs, steps):
    if method in ("logreg", "sklearn_mlp"):
        obj = LogisticRegression(C=1.0, max_iter=2000) if method == "logreg" else \
            MLPClassifier(hidden_layer_sizes=(128, 128, 64), max_iter=500, random_state=seed)
        obj.fit(xs, ys)
        raw = obj.predict_proba(xt)
        pr = np.zeros((len(xt), len(classes)))
        for j, c in enumerate(obj.classes_):
            pr[:, list(classes).index(c)] = raw[:, j]
        return np.asarray(classes)[pr.argmax(1)], pr, {"iterations": int(np.max(obj.n_iter_))}, None
    model, details = train_network(xs, ys, xc, yc, method, seed, classes, epochs, steps)
    p, pr, f = network_output(model, xt, classes)
    return p, pr, details, model


def _source_mask(task, batch):
    """Return the source-batch boolean mask for the current protocol family."""
    b = batch
    if task["family"] == "forward":
        return b < task["fold"]
    if task["family"] == "singlesource":
        return b == int(task["setting"][1:])  # setting stores the fixed source batch
    return b != task["fold"]


def _methods_for(family):
    if family == "closed":
        return MAIN_METHODS
    if family == "singlesource":
        return SINGLE_SOURCE_METHODS
    if family in ("osr", "gsa", "gsa_cv", "aq"):
        return ["base", "cal_sup", "cg_sup"]
    return ["base", "cal_sup", "cg_sup"]  # budget / forward


def run_closed(task, data_dir, out, epochs):
    from .data import load_drift
    x, y, b = load_drift(data_dir)
    target = np.flatnonzero(b == task["fold"])
    seed = task["seed"]
    source = np.flatnonzero(_source_mask(task, b))
    pool, test = stratified_parts(target, y, [0.2, 0.8], seed)
    pool = shuffled_pool(pool, seed)
    cal = nested_calibration(pool, target, y, task["budget"])
    check_disjoint(source=source, calibration=cal, test=test)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    methods = _methods_for(task["family"])
    r = Recorder(task, out)
    r.split(source_train=source, calibration_train=cal, test=test,
            unused_pool=np.setdiff1d(pool, cal))
    csv({"feature": np.arange(x.shape[1]), "mean": sc.mean_, "scale": sc.scale_},
        out / "preprocessing.csv")
    base = None
    for m in methods:
        t0 = time.time()
        if m == "mean_shift":
            assert base is not None
            shift = xc.mean(0) - xs.mean(0)
            p, pr, _ = network_output(base, xt - shift, np.arange(1, 7))
            details = {"reused_model": "base"}
        else:
            p, pr, details, model = _neural_or_lr(xs, y[source], xc, y[cal], xt, m, seed,
                                                  np.arange(1, 7), epochs, steps)
            if m == "base":
                base = model
        r.metric(m, classification_metrics(y[test], p, pr, np.arange(1, 7)), n_test=len(test))
        r.predictions(m, test, y[test], p, pr)
        r.configrow(m, n_source=len(source),
                    n_target_labeled=len(cal) if m in ("cal_sup", "cg_mix", "cg_sup", "cg_shuffle") else 0,
                    n_target_unlabeled=len(cal) if m in ("dann", "mean_shift") else 0,
                    n_test=len(test), seconds=time.time() - t0, **details)
    r.finish()


def run_osr(task, data_dir, out, epochs):
    from .data import load_drift
    x, y, b = load_drift(data_dir)
    seed = task["seed"]
    target = np.flatnonzero(b == task["fold"])
    unknown = np.array([int(c) for c in task["setting"].split("-")])
    classes = np.array([c for c in range(1, 7) if c not in unknown])
    source = np.flatnonzero((b != task["fold"]) & np.isin(y, classes))
    train, sval = stratified_parts(source, y, [0.9, 0.1], seed + 33)
    pool, thpool, test = stratified_parts(target, y, [0.2, 0.2, 0.6], seed)
    cal = nested_calibration(shuffled_pool(pool, seed), target, y, 0.1)
    cal = cal[np.isin(y[cal], classes)]
    check_disjoint(source_train=train, source_threshold=sval, calibration_train=cal,
                   target_threshold=thpool, test=test)
    sc = StandardScaler().fit(x[train])
    xs = sc.transform(x[train]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    xv = sc.transform(x[sval]).astype(np.float32)
    xp = sc.transform(x[thpool]).astype(np.float32)
    k = np.isin(y[test], classes)
    kp = np.isin(y[thpool], classes)
    if not k.any() or not (~k).any():
        raise ValueError("No known or unknown test examples")
    r = Recorder(task, out)
    r.split(source_train=train, source_threshold=sval, calibration_train=cal,
            target_threshold=thpool, test=test, unused_pool=np.setdiff1d(pool, cal))
    csv({"feature": np.arange(x.shape[1]), "mean": sc.mean_, "scale": sc.scale_},
        out / "preprocessing.csv")
    thresholds, sampling = [], []
    for mode in ["base", "cal_sup", "cg_sup"]:
        model, details = train_network(xs, y[train], xc, y[cal], mode, seed, classes, epochs,
                                       math.ceil((len(train) + len(pool) + 240) / 128))
        p, pr, ft = network_output(model, xt, classes)
        _, _, fs = network_output(model, xs, classes)
        _, _, fv = network_output(model, xv, classes)
        pp, ppr, fp = network_output(model, xp, classes)
        params = distance_model(fs, y[train])
        dt, dv, dp = distances(ft, params), distances(fv, params), distances(fp, params)
        area, curve = oscr(y[test], p, dt, k)
        csv(curve, out / f"oscr_curve_{mode}.csv")
        r.metric(mode, {"auroc": roc_auc_score(k.astype(int), -dt), "oscr": area,
                        "known_accuracy": accuracy_score(y[test][k], p[k]),
                        "known_macro_f1": f1_score(y[test][k], p[k], labels=np.unique(y[test][k]),
                                                   average="macro", zero_division=0)})
        r.predictions(mode, test, y[test], p, pr, distance=dt, is_known=k.astype(int))
        csv({"sample_id": thpool, "distance": dp, "is_known": kp.astype(int),
             "y_true": y[thpool]}, out / f"threshold_scores_{mode}.csv")
        csv({"sample_id": sval, "distance": dv, "y_true": y[sval]},
            out / f"source_validation_scores_{mode}.csv")
        rules = {"source_q90": (quantile_threshold(dv, 0.9), len(dv), 0.0)}
        if kp.any():
            rules["known_reference_q90"] = (quantile_threshold(dp[kp], 0.9), int(kp.sum()), 0.0)
        n = (min(100, int(kp.sum()), int((~kp).sum() / 0.75)) // 4) * 4
        if n >= 4:
            rng = np.random.RandomState(seed + task["fold"] * 17)
            ki = rng.permutation(np.flatnonzero(kp))
            ui = rng.permutation(np.flatnonzero(~kp))
            for rho in [0.0, 0.25, 0.5, 0.75]:
                nu = int(n * rho)
                sel = np.r_[ki[:n - nu], ui[:nu]]
                mix = dp[sel]
                tail = mix[mix <= np.median(mix)]
                tag = f"{int(rho * 100):02d}"
                rules[f"mixed{tag}_full_q90"] = (quantile_threshold(mix, 0.9), n, rho)
                rules[f"mixed{tag}_tail_q90"] = (quantile_threshold(tail, 0.9), n, rho)
                if mode == "base":
                    sampling.extend({"contamination": rho, "sample_id": int(thpool[i]),
                                     "is_known": int(kp[i])} for i in sel)
        for name, (thr, nref, rho) in rules.items():
            r.metric(mode, rejection_metrics(y[test], p, dt, k, thr),
                     threshold_rule=name, n_test=len(test))
            thresholds.append({"method": mode, "threshold_rule": name, "threshold": thr,
                               "n_reference": nref, "unknown_fraction": rho})
        r.configrow(mode, n_source=len(train), n_calibration_train=len(cal),
                    n_threshold_pool=len(thpool), n_mixed=n,
                    known_classes=";".join(map(str, classes)),
                    unknown_classes=task["setting"], **details)
    csv(thresholds, out / "thresholds.csv")
    csv(sampling, out / "mixed_pool_membership.csv")
    r.finish()


def run_gsa(task, data_dir, out, epochs):
    from .data import load_gsa
    from sklearn.model_selection import StratifiedKFold
    x, y, con = load_gsa(data_dir)
    seed = task["seed"]
    rng = np.random.RandomState(seed)
    if task["family"] == "gsa":
        target = np.flatnonzero(con == task["fold"])
        train = np.flatnonzero(con != task["fold"])
    else:
        train, target = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))[task["fold"]]
    cal = np.concatenate([rng.choice(target[y[target] == c], size=min(2, len(target[y[target] == c]) - 1),
                                     replace=False) for c in np.unique(y[target])])
    test = np.setdiff1d(target, cal)
    check_disjoint(source_train=train, calibration=cal, test=test)
    sc = StandardScaler().fit(x[train])
    xs = sc.transform(x[train]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    r = Recorder(task, out)
    r.split(source_train=train, calibration=cal, test=test)
    csv({"feature": np.arange(x.shape[1]), "mean": sc.mean_, "scale": sc.scale_},
        out / "preprocessing.csv")
    for mode in ["base", "cal_sup", "cg_sup"]:
        xb, yb = xs.copy(), y[train].copy()
        if mode != "base":
            xb = np.r_[xb, xc]
            yb = np.r_[yb, y[cal]]
        if mode == "cg_sup":
            gen = np.random.RandomState(seed)
            xx, yy = [], []
            for c in np.unique(y[cal]):
                ids = np.flatnonzero(y[train] == c)
                center = 0.5 * xs[ids].mean(0) + 0.5 * xc[y[cal] == c].mean(0)
                for _ in range(10):
                    i = gen.choice(ids)
                    lam = gen.beta(0.5, 0.5) * 0.6
                    xx.append((1 - lam) * xs[i] + lam * center)
                    yy.append(c)
            xb = np.r_[xb, np.stack(xx)]
            yb = np.r_[yb, yy]
        model = LogisticRegression(C=1.0, max_iter=2000).fit(xb, yb)
        pr = model.predict_proba(xt)
        p = model.predict(xt)
        r.metric(mode, classification_metrics(y[test], p, pr, model.classes_), n_test=len(test))
        r.predictions(mode, test, y[test], p, pr)
        r.configrow(mode, n_source=len(train), n_calibration=len(cal) if mode != "base" else 0,
                    n_test=len(test), classifier="LogisticRegression", iterations=model.n_iter_[0])
    r.finish()


def _chronological_blocks(b):
    return {i: tuple(np.split(np.flatnonzero(b == i),
                              [int(0.6 * np.sum(b == i)), int(0.8 * np.sum(b == i))]))
            for i in range(4)}


def _aq_setup(data_dir, A, B, seed):
    from .data import load_aq
    x, c, b, rawids, times = load_aq(data_dir)
    blocks = _chronological_blocks(b)
    atr, ava, ate = blocks[A]
    btr, bva, bte = blocks[B]
    low = np.quantile(c[atr], 0.5, axis=0)
    high = np.quantile(c[atr], 0.95, axis=0)
    normal = (c <= low).all(1)
    event = (c > high).any(1)
    keep = normal | event
    rng = np.random.RandomState(seed)
    cn = btr[normal[btr]]
    ct = bva[normal[bva]]
    if min(len(cn), len(ct)) < 3:
        raise ValueError(f"insufficient normal reference: {len(cn)}, {len(ct)}")
    cn = rng.choice(cn, min(15, len(cn)), replace=False)
    ct = rng.choice(ct, min(15, len(ct)), replace=False)
    train = np.setdiff1d(atr[keep[atr]], np.r_[cn, ct])
    val = np.setdiff1d(ava[normal[ava]], np.r_[cn, ct])
    test = bte[keep[bte]]
    broad = bte
    if len(np.unique(event[test])) < 2 or len(val) < 3 or len(np.unique(event[train])) < 2:
        raise ValueError("insufficient source or test label support")
    check_disjoint(source_train=train, source_threshold=val, target_train_calibration=cn,
                   target_threshold_calibration=ct, test=broad)
    return (x, c, b, rawids, times, train, val, cn, ct, test, broad, normal, event, low, high)


def run_aq(task, data_dir, out, epochs):
    A, B = map(int, task["setting"].split("-"))
    seed = task["seed"]
    (x, c, b, rawids, times, tr, sv, cn, ct, te, broad, normal, event, low, high) = \
        _aq_setup(data_dir, A, B, seed)
    sc = StandardScaler().fit(x[tr])
    xs = sc.transform(x[tr]).astype(np.float32)
    xv = sc.transform(x[sv]).astype(np.float32)
    xc = sc.transform(x[cn]).astype(np.float32)
    xct = sc.transform(x[ct]).astype(np.float32)
    xb = sc.transform(x[broad]).astype(np.float32)
    yt = event[tr].astype(int)
    in_strict = np.isin(broad, te)
    r = Recorder(task, out)
    r.split(source_train=rawids[tr], source_threshold=rawids[sv],
            target_train_calibration=rawids[cn], target_threshold_calibration=rawids[ct],
            test=rawids[broad])
    csv({"feature": np.arange(x.shape[1]), "mean": sc.mean_, "scale": sc.scale_},
        out / "preprocessing.csv")
    csv({"reference": ["CO", "C6H6", "NOx", "NO2"], "normal_p50": low, "event_p95": high},
        out / "label_thresholds.csv")
    scores, thresholds = {}, []
    pca = PCA(n_components=0.95).fit(xs[yt == 0])
    recon = lambda z: np.square(z - pca.inverse_transform(pca.transform(z))).sum(1)
    scores["pca"] = (recon(xb), quantile_threshold(recon(xv), 0.95))
    steps = math.ceil((len(tr) + 30 + 40) / 128)
    for mode in ["base", "cal_sup", "cg_sup"]:
        model, details = train_network(xs, yt, xc, np.zeros(len(cn), int), mode, seed, [0, 1],
                                       epochs=80 if epochs == 50 else epochs, steps=steps, small=True)
        _, _, fs = network_output(model, xs, [0, 1])
        _, _, fv = network_output(model, xv, [0, 1])
        _, _, fc = network_output(model, xc, [0, 1])
        _, _, fct = network_output(model, xct, [0, 1])
        _, _, fb = network_output(model, xb, [0, 1])
        params = distance_model(fs[yt == 0], np.zeros(np.sum(yt == 0), int))
        shift = fc.mean(0) - params[0][0]
        ds = distances(fb, params)
        dct = distances(fct, params)
        dsv = distances(fv, params)
        shifted = distances(fb - shift, params)
        shiftedct = distances(fct - shift, params)
        src_thr = quantile_threshold(dsv, 0.95)
        tgt_thr = quantile_threshold(dct, 0.95)
        shift_thr = quantile_threshold(shiftedct, 0.95)
        if mode == "base":
            scores.update({"mlp": (ds, src_thr), "threshold_only": (ds, tgt_thr),
                           "shift_only": (shifted, src_thr), "shift_threshold": (shifted, shift_thr)})
        elif mode == "cal_sup":
            scores.update({"cal_sup": (ds, src_thr), "cal_sup_threshold": (ds, tgt_thr)})
        else:
            scores.update({"cg_train": (ds, src_thr), "cg_train_threshold": (ds, tgt_thr),
                           "cg_full": (shifted, shift_thr)})
        csv({"sample_id": rawids[ct], "distance_unshifted": dct, "distance_shifted": shiftedct},
            out / f"threshold_scores_{mode}.csv")
        csv({"sample_id": rawids[sv], "distance": dsv}, out / f"source_validation_scores_{mode}.csv")
        csv({"feature": np.arange(len(shift)), "shift": shift}, out / f"feature_shift_{mode}.csv")
        r.configrow(mode, n_source=len(tr), n_source_normal=int(np.sum(yt == 0)),
                    n_calibration_training=len(cn), n_calibration_threshold=len(ct),
                    n_strict_test=len(te), n_all_test=len(broad), source_domain=A, target_domain=B,
                    **details)
    for method, (d, thr) in scores.items():
        for name, mask in [("extreme", in_strict),
                           ("including_intermediate", np.ones(len(broad), bool))]:
            y = event[broad][mask].astype(int)
            r.metric(method, binary_metrics(y, d[mask], thr), evaluation=name, n_test=int(mask.sum()))
        r.predictions(method, rawids[broad], event[broad].astype(int), (d > thr).astype(int),
                      distance=d, in_extreme_task=in_strict.astype(int), timestamp=times[broad],
                      threshold=thr)
        thresholds.append({"method": method, "threshold": thr})
    csv(thresholds, out / "thresholds.csv")
    r.finish()


RUNNERS = {
    "closed": run_closed, "budget": run_closed, "forward": run_closed,
    "singlesource": run_closed, "osr": run_osr, "gsa": run_gsa, "gsa_cv": run_gsa,
    "aq": run_aq,
}


# ---------------------------------------------------------------------------
# Task planning and execution
# ---------------------------------------------------------------------------
def task_list(seeds=None, families=None, extra=None):
    """Build the full planned task list (optionally filtered by family).

    Parameters
    ----------
    seeds : list of int, optional
        Defaults to :data:`gasbench.utils.SEEDS`.
    families : iterable of str, optional
        Subset of families to include.
    extra : list of dict, optional
        Additional ad-hoc tasks (each with family/setting/budget/seed/fold).
    """
    seeds = seeds or SEEDS
    tasks = []

    def add(family, setting, budget, folds):
        for seed in seeds:
            for fold in folds:
                tasks.append(dict(family=family, setting=setting, budget=budget,
                                  seed=seed, fold=fold))

    add("closed", "lobo", 0.1, range(1, 11))
    for frac in [0.05, 0.2]:
        add("budget", "lobo", frac, range(1, 11))
    add("forward", "past_to_future", 0.1, range(2, 11))
    add("singlesource", "b1", 0.1, range(2, 11))
    for u in ["4-6", "1-2", "3-5"]:
        add("osr", u, 0.1, range(1, 11))
    add("gsa", "concentration", 2, [50, 100, 200])
    add("gsa_cv", "stratified_cv", 2, range(5))
    for A in range(4):
        for B in range(4):
            add("aq", f"{A}-{B}", 30, [0])
    if extra:
        tasks.extend(extra)
    if families is not None:
        wanted = set(families)
        tasks = [t for t in tasks if t["family"] in wanted]
    return tasks


def task_id(t):
    return f"{t['family']}_{t['setting']}_b{t['budget']}_s{t['seed']}_f{t['fold']}"


def _code_hash(package_root):
    h = hashlib.sha256()
    for mod in ["utils.py", "evaluate.py", "methods.py", "data.py", "protocols.py"]:
        h.update(sha(Path(package_root) / mod).encode())
    return h.hexdigest()


def execute_task(task, data_dir, root, epochs=50):
    """Run one task (cached and hashed). Returns (task_id, status, seconds)."""
    out = Path(root) / "tasks" / task_id(task)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "completion.csv"
    import gasbench as gb
    digest = _code_hash(Path(gb.__file__).parent)
    if done.exists():
        old = pd.read_csv(done)
        if old.iloc[0]["code_sha256"] != digest:
            raise RuntimeError("Code changed after completed task; choose new output root")
        return task_id(task), "cached", float(old.iloc[0].seconds)
    started = time.time()
    csv([{**task, "code_sha256": digest, "epochs": epochs}], out / "task_config.csv")
    try:
        RUNNERS[task["family"]](task, data_dir, out, epochs)
        csv([{"status": "complete", "code_sha256": digest, "seconds": time.time() - started}], done)
        return task_id(task), "complete", time.time() - started
    except Exception as e:
        csv([{"status": "failed", "error": repr(e), "traceback": traceback.format_exc(),
              "seconds": time.time() - started}], out / "failure.csv")
        if task["family"] == "aq" and "insufficient" in str(e):
            csv([{"status": "ineligible", "reason": str(e), "code_sha256": digest,
                  "seconds": time.time() - started}], out / "ineligible.csv")
            return task_id(task), "ineligible", time.time() - started
        return task_id(task), "failed", time.time() - started


def aggregate(root):
    """Collate per-task CSV metrics into the standard summary files."""
    root = Path(root)
    frames, status = [], []
    for t in task_list():
        path = root / "tasks" / task_id(t)
        complete = (path / "completion.csv").exists()
        status.append({**t, "task_id": task_id(t),
                       "status": "complete" if complete else
                       "ineligible" if (path / "ineligible.csv").exists() else
                       "failed" if (path / "failure.csv").exists() else "pending"})
        if complete:
            frames.append(pd.read_csv(path / "metrics.csv"))
    csv(status, root / "task_status.csv")
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    csv(data, root / "metrics_by_fold_seed.csv")
    per, agg = seed_summary(data, KEYS)
    csv(per, root / "metrics_by_seed.csv")
    csv(agg, root / "summary_mean_sd.csv")
    _, fold = seed_summary(data, KEYS + ["fold"])
    csv(fold, root / "summary_by_fold_mean_sd.csv")
    aq = data[data.family == "aq"].copy()
    if len(aq):
        aq["setting"] = aq.setting.map(
            lambda v: "same_domain" if v.split("-")[0] == v.split("-")[1] else "cross_domain")
        a, b = seed_summary(aq, KEYS)
        csv(a, root / "aq_domain_metrics_by_seed.csv")
        csv(b, root / "aq_domain_summary_mean_sd.csv")
    diffs = []
    for family, base, comparison in [("closed", "base", "cg_sup"),
                                     ("closed", "cal_sup", "cg_sup"),
                                     ("singlesource", "base", "cg_sup"),
                                     ("singlesource", "cal_sup", "cg_sup"),
                                     ("gsa", "cal_sup", "cg_sup"),
                                     ("gsa", "base", "cal_sup")]:
        sub = data[(data.family == family) & (data.threshold_rule == "none")]
        ix = ["family", "setting", "budget", "seed", "fold", "evaluation", "metric"]
        wide = sub.pivot_table(index=ix, columns="method", values="value").reset_index()
        if base in wide and comparison in wide:
            wide["value"] = wide[comparison] - wide[base]
            wide["method"] = f"{comparison}-minus-{base}"
            wide["threshold_rule"] = "none"
            diffs.append(wide[ix + ["value", "method", "threshold_rule"]])
    if diffs:
        d = pd.concat(diffs)
        p, a = seed_summary(d, KEYS)
        csv(d, root / "paired_differences_by_fold.csv")
        csv(p, root / "paired_differences_by_seed.csv")
        csv(a, root / "paired_differences_mean_sd.csv")


def freeze(data_dir, root):
    """Write data manifest and environment info for reproducibility."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    import gasbench as gb
    from .data import KNOWN_SHA
    rows = []
    for name, ref in KNOWN_SHA.items():
        p = Path(data_dir) / ("processed" if name == "drift_clean.csv" else "raw") / name
        rows.append({"file": str(p), "sha256": sha(p) if p.exists() else None,
                     "bytes": p.stat().st_size if p.exists() else None,
                     "matches_paper_hash": (sha(p) == ref) if p.exists() else None})
    csv(rows, root / "data_manifest.csv")
    csv([{"protocol": PROTOCOL_VERSION, "python": platform.python_version(),
          "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
          "sklearn": sklearn.__version__, "platform": platform.platform(),
          "code_sha256": _code_hash(Path(gb.__file__).parent),
          "seeds": ";".join(map(str, SEEDS)), "std_ddof": 1}],
        root / "environment.csv")
    csv(task_list(), root / "planned_tasks.csv")
