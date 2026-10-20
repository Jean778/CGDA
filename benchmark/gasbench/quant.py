"""INT8 fake-quantization and structured pruning primitives for the CGDA
deployment coupling study.

Design goal: answer "does INT8 quantization / structured pruning break the
calibration-guided training gains?" Under the exact closed LOBO protocol of
revision_v2, the study compares:

* fp32        -- cg_sup training, FP32 inference (authoritative baseline)
* ptq         -- trained FP32 weights, post-training INT8 fake-quant (no retrain)
* ptq_ft      -- PTQ + a short source-domain fine-tune of the quantized net
* qat         -- cg_sup training from scratch with fake-quant in the loop
* prune30_ptq -- structured 30% neuron pruning + PTQ, no retrain
* prune30_qat -- structured 30% neuron pruning + quantized fine-tuning
                 (deployment-time prune + quantize + on-site recalibration)

Fake quantization is symmetric per-tensor INT8 with a straight-through
estimator (gradients flow through rounding as identity), matching the
PTQ/QAT numerics of CMSIS-NN-style deployments.

fp32 itself is NOT implemented here: reuse revision_core.train_network so the
authoritative numbers stay untouched.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["fake_quant", "QuantLinear", "QuantNetwork",
           "structured_prune", "train_quant", "network_output"]


def fake_quant(x: torch.Tensor, bits: int = 8,
               scale: float | np.ndarray | None = None) -> torch.Tensor:
    """Symmetric per-tensor (or per-feature) fake quant, STE gradient.

    * scale=None      -> per-tensor max|w| scale (weights only)
    * scale=float     -> fixed scalar scale (activations, calibration-derived)
    * scale=ndarray   -> per-feature fixed scale (input channels)
    Long-tailed gas features make per-batch max scales collapse, so
    activations must use calibration-derived fixed scales.
    """
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = x.abs().max().clamp(min=1e-8) / qmax
    else:
        scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    x = x.clamp(-qmax * scale, qmax * scale)
    q = torch.round(x / scale).clamp(-qmax, qmax) * scale
    return x + (q - x).detach()


class QuantLinear(nn.Module):
    """Linear with INT8 fake-quantized weight.

    Input is quantized only when `in_scale` is given (calibration-derived
    fixed scalar or per-feature scale); hidden activations stay FP32, which
    matches the common mixed-precision TinyML deployment (INT8 weights +
    FP32 input features).
    """

    def __init__(self, in_f: int, out_f: int, bits: int = 8,
                 in_scale: float | np.ndarray | None = None):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.bits = bits
        self.in_scale = in_scale

    def forward(self, x):
        if self.in_scale is not None:
            x = fake_quant(x, self.bits, self.in_scale)
        w = fake_quant(self.fc.weight, self.bits)
        return F.linear(x, w, self.fc.bias)


class QuantNetwork(nn.Module):
    """Same architecture as revision_core.Network but with fake-quant weights.

    [128 -> 128 -> 128 -> 64 -> 6] with ReLU; only the input layer quantizes
    its input (fixed calibration scale), hidden activations remain FP32.
    """

    def __init__(self, dim: int, nclasses: int, bits: int = 8,
                 in_scale: float | None = None, widths=(128, 128, 64)):
        super().__init__()
        self.bits = bits
        self.fe = nn.Sequential(
            QuantLinear(dim, widths[0], bits, in_scale), nn.ReLU(),
            QuantLinear(widths[0], widths[1], bits), nn.ReLU(),
            QuantLinear(widths[1], widths[2], bits), nn.ReLU(),
        )
        self.cls = QuantLinear(widths[2], nclasses, bits)

    def forward(self, x):
        return self.cls(self.fe(x))


def structured_prune(model: nn.Module, rate: float):
    """Structured output-neuron pruning by outgoing-weight L2 norm.

    Keeps ceil((1-rate)*width) output units per hidden layer and rebuilds the
    network with cascaded input dims. Returns a lightweight FP32 module with
    `.fe` (Sequential of Linear) and `.cls` (Linear), matching the interface
    of revision_core.Network. Deterministic for a given trained model.
    """
    if rate <= 0:
        return model
    keep = lambda w: int(np.ceil(w * (1 - rate)))  # noqa: E731
    w1 = model.fe[0].weight.detach().numpy()
    w2 = model.fe[2].weight.detach().numpy()
    w3 = model.fe[4].weight.detach().numpy()
    w4 = model.cls.weight.detach().numpy()

    idx1 = np.argsort(-np.linalg.norm(w1, axis=1))[:keep(128)]
    w1p, b1p = w1[idx1], model.fe[0].bias.detach().numpy()[idx1]
    w2p = w2[:, idx1]
    idx2 = np.argsort(-np.linalg.norm(w2p, axis=1))[:keep(128)]
    w2p, b2p = w2p[idx2], model.fe[2].bias.detach().numpy()[idx2]
    w3p = w3[:, idx2]
    idx3 = np.argsort(-np.linalg.norm(w3p, axis=1))[:keep(64)]
    w3p, b3p = w3p[idx3], model.fe[4].bias.detach().numpy()[idx3]
    w4p = w4[:, idx3]

    net = nn.Sequential(
        nn.Linear(w1.shape[1], len(idx1)), nn.ReLU(),
        nn.Linear(len(idx1), len(idx2)), nn.ReLU(),
        nn.Linear(len(idx2), len(idx3)), nn.ReLU(),
    )
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(w1p)); net[0].bias.copy_(torch.from_numpy(b1p))
        net[2].weight.copy_(torch.from_numpy(w2p)); net[2].bias.copy_(torch.from_numpy(b2p))
        net[4].weight.copy_(torch.from_numpy(w3p)); net[4].bias.copy_(torch.from_numpy(b3p))
    cls = nn.Linear(len(idx3), w4.shape[0])
    with torch.no_grad():
        cls.weight.copy_(torch.from_numpy(w4p))
        cls.bias.copy_(model.cls.bias.detach())

    out = nn.Module()
    out.fe = net
    out.cls = cls
    return out


def _load_into_quant(model: QuantNetwork, src) -> None:
    """Copy an FP32 (possibly pruned) module's weights into QuantNetwork."""
    with torch.no_grad():
        model.fe[0].fc.weight.copy_(src.fe[0].weight)
        model.fe[0].fc.bias.copy_(src.fe[0].bias)
        model.fe[2].fc.weight.copy_(src.fe[2].weight)
        model.fe[2].fc.bias.copy_(src.fe[2].bias)
        model.fe[4].fc.weight.copy_(src.fe[4].weight)
        model.fe[4].fc.bias.copy_(src.fe[4].bias)
        model.cls.fc.weight.copy_(src.cls.weight)
        model.cls.fc.bias.copy_(src.cls.bias)


def train_quant(xs, ys, xc, yc, seed, classes, epochs=50, steps=None, bits=8,
                quantize=False, prune=0.0, ft_epochs=0, pretrained=None,
                in_scale: float | None = None):
    """cg_sup-style training (centroid Mixup + calibration supervision) with
    optional fake-quant / structured pruning / fine-tuning.

    * no pretrained            : train from scratch (qat when quantize=True)
    * pretrained + ft_epochs=0 : evaluate quantized/pruned weights as-is (PTQ)
    * pretrained + ft_epochs>0 : quantize (and prune) then fine-tune on the
                                 source domain (ptq_ft / prune30_qat)

    `in_scale` is the calibration-derived fixed input scale for INT8 input
    quantization; pass it for quantized configs (computed from the calibration
    pool by the caller), None keeps the input in FP32.
    Returns (model, details).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    classes = np.asarray(classes)
    lut = {int(c): i for i, c in enumerate(classes)}
    ys = np.array([lut[int(v)] for v in ys])
    yc = np.array([lut[int(v)] for v in yc])

    src = pretrained
    if src is not None and prune > 0:
        src = structured_prune(src, prune)

    if quantize:
        if src is not None:
            widths = (src.fe[0].weight.shape[0], src.fe[2].weight.shape[0],
                      src.fe[4].weight.shape[0])
            model = QuantNetwork(src.fe[0].weight.shape[1],
                                 src.cls.weight.shape[0], bits=bits,
                                 in_scale=in_scale, widths=widths)
            _load_into_quant(model, src)
        else:
            model = QuantNetwork(xs.shape[1], len(classes), bits=bits,
                                 in_scale=in_scale)
    else:
        # non-quantized path used only by pruning experiments without PTQ
        model = src if src is not None else _scratch_fp32(xs.shape[1], len(classes))

    n_epochs = ft_epochs if (pretrained is not None and ft_epochs > 0) else epochs
    if n_epochs <= 0:
        return model, {"epochs": 0, "optimizer_steps": 0,
                       "parameter_count": _numel(model), "bits": bits,
                       "quantize": quantize, "prune": prune,
                       "class_weights": ""}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    original = {1: 1., 2: 1., 3: 1., 4: 1.5, 5: 1., 6: 2.}
    weights = np.array([original.get(int(c), 1.) for c in classes], np.float32)
    wt = torch.from_numpy(weights)

    xb = np.concatenate([xs, xc]) if len(xc) else xs
    yb = np.r_[ys, yc] if len(xc) else ys
    centers = {c: xc[yc == c].mean(0) for c in np.unique(yc)}
    batch = 128
    steps = steps or math.ceil((len(xs) + len(xc) + len(classes) * 60) / batch)

    for epoch in range(n_epochs):
        xa, ya = xb, yb
        if centers:
            xx, yy = [], []
            for c, center in centers.items():
                count = 60
                ids = rng.choice(np.flatnonzero(yb == c), count, replace=True)
                lam = rng.beta(.5, .5, (count, 1)).astype(np.float32)
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
            loss = F.cross_entropy(model(x), y, weight=wt)
            loss.backward()
            optimizer.step()
    return model, {"epochs": n_epochs, "optimizer_steps": steps * n_epochs,
                   "parameter_count": _numel(model), "bits": bits,
                   "quantize": quantize, "prune": prune,
                   "class_weights": ";".join(map(str, weights))}


def _scratch_fp32(dim: int, nclasses: int):
    net = nn.Sequential(
        nn.Linear(dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
    )
    cls = nn.Linear(64, nclasses)
    out = nn.Module()
    out.fe = net
    out.cls = cls
    return out


def _numel(model) -> int:
    return sum(p.numel() for p in model.parameters())


def network_output(model, x, classes):
    model.eval()
    with torch.no_grad():
        f = model.fe(torch.from_numpy(np.asarray(x, np.float32)))
        probs = torch.softmax(model.cls(f), 1).numpy()
    return np.asarray(classes)[probs.argmax(1)], probs
