"""Model implementations for GasBench.

All neural methods share the same lightweight MLP backbone (128-128-64) and
the same number of optimizer updates, so performance differences come from the
training mechanism, not from capacity or compute. New methods can be added by
writing a function `(xs, ys, xc, yc, seed, classes, epochs, steps) -> model,
details` and registering it with :func:`register_method`.
"""
import math

import numpy as np
import torch
from torch import nn

# torch.set_num_threads(1) and deterministic mode are set in cli/protocols to
# keep them configurable; this module must not hard-code them on import.


class Network(nn.Module):
    """Lightweight MLP feature extractor + linear head."""

    def __init__(self, dim, nclasses, small=False):
        super().__init__()
        widths = [32, 16] if small else [128, 128, 64]
        layers = []
        prev = dim
        for w in widths:
            layers.extend([nn.Linear(prev, w), nn.ReLU()])
            prev = w
        self.fe = nn.Sequential(*layers)
        self.cls = nn.Linear(prev, nclasses)

    def forward(self, x):
        return self.cls(self.fe(x))


class GRL(torch.autograd.Function):
    """Gradient reversal layer used by the DANN comparator."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


DEFAULT_WEIGHTS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.5, 5: 1.0, 6: 2.0}


def train_network(xs, ys, xc, yc, mode, seed, classes, epochs=50, steps=None, small=False):
    """Train a backbone under the given mechanism with equal optimizer budgets.

    Parameters
    ----------
    xs, ys : arrays
        Source features and labels.
    xc, yc : arrays
        Calibration features and labels (empty for methods that do not use them).
    mode : str
        One of {"base", "cal_sup", "cg_mix", "cg_sup", "cg_shuffle",
        "mixup_std", "dann"}.
    seed, classes, epochs, steps, small : see protocol runners.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    classes = np.asarray(classes)
    lut = {int(c): i for i, c in enumerate(classes)}
    ys = np.array([lut[int(v)] for v in ys])
    yc = np.array([lut[int(v)] for v in yc])
    model = Network(xs.shape[1], len(classes), small)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005 if small else 0.001)
    if mode == "dann":
        domain = nn.Linear(16 if small else 64, 1)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(domain.parameters()), lr=0.001)

    weights = np.ones(len(classes), np.float32)
    if not small:
        weights = np.array([DEFAULT_WEIGHTS.get(int(c), 1.0) for c in classes], np.float32)
    wt = torch.from_numpy(weights)

    use_cal = mode in ("cal_sup", "cg_sup", "cg_shuffle")
    xb = np.concatenate([xs, xc]) if use_cal and len(xc) else xs
    yb = np.r_[ys, yc] if use_cal and len(xc) else ys
    centers = {c: xc[yc == c].mean(0) for c in np.unique(yc)}
    if mode == "cg_shuffle" and len(centers) > 1:
        keys = sorted(centers)
        centers = {c: centers[keys[(i + 1) % len(keys)]] for i, c in enumerate(keys)}

    batch = 128
    steps = steps or math.ceil((len(xs) + len(xc) + len(classes) * 60) / batch)
    for epoch in range(epochs):
        xa, ya = xb, yb
        if mode in ("cg_mix", "cg_sup", "cg_shuffle") and centers:
            xx, yy = [], []
            for c, center in centers.items():
                count = 40 if small else 60
                ids = rng.choice(np.flatnonzero(yb == c), count, replace=True)
                lam = rng.beta(0.5, 0.5, (count, 1)).astype(np.float32)
                xx.append((1 - lam) * xb[ids] + lam * center)
                yy.append(np.full(count, c))
            xa = np.concatenate([xb, *xx])
            ya = np.concatenate([yb, *yy])
        xt = torch.from_numpy(np.asarray(xa, np.float32))
        yt = torch.from_numpy(ya).long()
        order = np.resize(rng.permutation(len(xt)), steps * batch)
        for i in range(steps):
            ids = order[i * batch:(i + 1) * batch]
            x, y = xt[ids], yt[ids]
            optimizer.zero_grad(set_to_none=True)
            if mode == "mixup_std":
                j = rng.permutation(len(x))
                lam = float(rng.beta(0.5, 0.5))
                logits = model(lam * x + (1 - lam) * x[j])
                loss = lam * nn.functional.cross_entropy(logits, y, weight=wt) + \
                    (1 - lam) * nn.functional.cross_entropy(logits, y[j], weight=wt)
            elif mode == "dann":
                h = model.fe(x)
                logits = model.cls(h)
                loss = nn.functional.cross_entropy(logits, y, weight=wt)
                if len(xc):
                    target = torch.from_numpy(xc[rng.randint(len(xc), size=batch)])
                    ds = domain(GRL.apply(h, 0.3))
                    dt = domain(GRL.apply(model.fe(target), 0.3))
                    loss = loss + 0.5 * (
                        nn.functional.binary_cross_entropy_with_logits(ds, torch.ones_like(ds))
                        + nn.functional.binary_cross_entropy_with_logits(dt, torch.zeros_like(dt))
                    )
            else:
                loss = nn.functional.cross_entropy(model(x), y, weight=wt)
            loss.backward()
            optimizer.step()
    return model, {
        "optimizer_steps": steps * epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "batch_size": batch,
        "class_weights": ";".join(map(str, weights)),
    }


def network_output(model, x, classes):
    """Predictions, class probabilities and penultimate features."""
    model.eval()
    with torch.no_grad():
        f = model.fe(torch.from_numpy(np.asarray(x, np.float32)))
        probs = torch.softmax(model.cls(f), 1).numpy()
    return np.asarray(classes)[probs.argmax(1)], probs, f.numpy()


def distance_model(f, y):
    """Per-class Gaussian (mean, pinv-cov) fitted on the given features."""
    params = []
    for c in np.unique(y):
        z = f[y == c].astype(float)
        mu = z.mean(0)
        cov = np.cov(z, rowvar=False) + np.eye(z.shape[1]) * 1e-6
        params.append((mu, np.linalg.pinv(cov, hermitian=True)))
    return params


def distances(f, params):
    """Minimum Mahalanobis distance of each sample to any known class."""
    out = []
    for mu, inv in params:
        diff = f.astype(float) - mu
        out.append(np.maximum(0, np.einsum("ij,jk,ik->i", diff, inv, diff)))
    return np.min(out, axis=0)


# ---------------------------------------------------------------------------
# Method registry (extensible)
# ---------------------------------------------------------------------------
_METHODS = {"base", "cal_sup", "cg_mix", "cg_sup", "cg_shuffle", "mixup_std", "dann"}


def list_methods():
    """Return the names of all available training mechanisms."""
    return sorted(_METHODS)


def register_method(name):
    """Decorator to register an extra training mechanism.

    The wrapped function must have the signature
    ``(xs, ys, xc, yc, seed, classes, epochs, steps, small) -> (model, details)``
    and return a ``Network``-like object exposing ``fe`` and ``cls`` so that
    :func:`network_output` works unchanged.
    """
    def deco(fn):
        _METHODS.add(name)
        return fn
    return deco
