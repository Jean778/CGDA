"""Shared evaluation primitives for GasBench.

No project data are read on import; all functions are deterministic given
their inputs. Thread limits are set here so that every entry point inherits
the same CPU budget.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

#: Default random seeds used by every protocol (5 seeds, matched to the paper).
SEEDS = [42, 2024, 7, 123, 2025]


def check_disjoint(**sets):
    """Assert that the given index sets are internally unique and pairwise disjoint."""
    keys = list(sets)
    for i, a in enumerate(keys):
        arr = np.asarray(sets[a])
        assert len(arr) == len(np.unique(arr)), f"duplicate indices in {a}"
        for b in keys[i + 1:]:
            assert not np.intersect1d(np.asarray(sets[a]), np.asarray(sets[b])).size, f"overlap: {a}, {b}"


def stratified_parts(indices, y, fractions, seed):
    """Partition within observed classes, preserving input identity and minimum counts.

    Parameters
    ----------
    indices : array of int
        Sample indices of the domain to partition.
    y : array of int
        Class labels (same length as the *whole* dataset, indexed by `indices`).
    fractions : list of float
        Non-negative proportions; last fraction receives the remainder.
    seed : int
        Random seed for the within-class shuffle.

    Returns
    -------
    tuple of arrays : one index array per fraction.
    """
    rng = np.random.RandomState(seed)
    out = [[] for _ in fractions]
    for c in np.unique(y[indices]):
        z = indices[y[indices] == c].copy()
        rng.shuffle(z)
        counts = np.floor(np.array(fractions[:-1]) * len(z)).astype(int)
        # Tiny classes: reserve evaluation samples rather than fabricate coverage.
        if len(z) >= len(fractions):
            counts = np.maximum(counts, 1)
            while counts.sum() >= len(z):
                counts[np.argmax(counts)] -= 1
        cur = 0
        for i, n in enumerate(counts):
            out[i].extend(z[cur:cur + n])
            cur += n
        out[-1].extend(z[cur:])
    return tuple(np.array(sorted(z), dtype=int) for z in out)


def nested_calibration(pool, target, y, fraction):
    """Deterministic per-class calibration selection from a pre-shuffled pool.

    The caller must supply a pool already randomised by the seed; this function
    never inspects feature values or model outputs, eliminating selection bias.

    Parameters
    ----------
    pool : array of int
        Calibration pool indices (already shuffled).
    target : array of int
        All target-domain indices (used only for per-class counts).
    y : array of int
        Labels indexed by dataset position.
    fraction : float
        Budget fraction relative to the per-class target size.
    """
    ans = []
    for c in np.unique(y[target]):
        z = pool[y[pool] == c]
        n = min(len(z), max(1, int(fraction * np.sum(y[target] == c))))
        ans.extend(z[:n])
    return np.array(ans, dtype=int)
