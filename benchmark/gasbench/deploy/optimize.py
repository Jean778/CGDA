"""Hardware-aware optimization simulation: INT8 post-training quantization
and structured (neuron-level) pruning.

Both transformations are modelled analytically on the MLP cost profile:

* INT8 PTQ        : weight/activation bytes -> 1/4 of FP32; MCU/SBC get the
                    platform `int8_speedup`; Edge TPU requires INT8.
* structured prune: removing an output neuron of layer L kills one column of
                    W_L and one row of W_{L+1}; MACs drop by the pruned
                    fraction on both layers (with the head kept dense).

Accuracy impact is NOT measured here; we report the published-order estimate
(<1% loss for <=30% prune / INT8 PTQ on tiny MLPs) and mark it as an estimate
so the optimization curve is read as a latency/energy trade-off, not a
measured accuracy claim.
"""
from __future__ import annotations

from .cost_model import ModelProfile, LayerCost, BYTES, profile_report


def quantize_profile(prof: ModelProfile) -> ModelProfile:
    """Return the INT8 variant of the profile (same MACs, 1/4 memory)."""
    out = ModelProfile(dim=prof.dim, widths=prof.widths, nclasses=prof.nclasses,
                       dtype="int8")
    out.layers = [
        LayerCost(name=l.name, in_features=l.in_features, out_features=l.out_features,
                  macs=l.macs, parameters=l.parameters,
                  weight_bytes=l.weight_bytes // 4,
                  activation_bytes=l.activation_bytes // 4)
        for l in prof.layers
    ]
    out.total_macs = prof.total_macs
    out.total_parameters = prof.total_parameters
    out.weight_bytes = sum(l.weight_bytes for l in out.layers)
    out.peak_activation_bytes = max(l.activation_bytes for l in out.layers)
    return out


def prune_profile(prof: ModelProfile, rate: float) -> ModelProfile:
    """Structured neuron pruning: drop `rate` fraction of hidden neurons.

    Output neurons of fc layers (not the head) are pruned; MACs and memory of
    the affected layer and its successor shrink accordingly. Head stays dense.
    """
    assert 0.0 <= rate < 1.0
    keep = [1.0 - rate] * len(prof.widths)  # per-hidden-layer keep fraction
    layers = []
    total_macs = 0
    total_params = 0
    prev_in = prof.dim
    for idx, l in enumerate(prof.layers):
        # Hidden fc layers keep (1-rate) of outputs; the head keeps all.
        is_hidden = idx < len(prof.widths)
        out = max(1, int(round(l.out_features * keep[idx]))) if is_hidden else l.out_features
        # Input dimension is the *kept* output of the previous layer.
        macs = prev_in * out
        params = prev_in * out + out
        layers.append(LayerCost(name=l.name, in_features=prev_in, out_features=out,
                                macs=macs, parameters=params,
                                weight_bytes=params * BYTES[prof.dtype],
                                activation_bytes=out * BYTES[prof.dtype]))
        total_macs += macs
        total_params += params
        prev_in = out
    out = ModelProfile(dim=prof.dim, widths=[int(round(w * (1 - rate))) for w in prof.widths],
                       nclasses=prof.nclasses, layers=layers, dtype=prof.dtype)
    out.total_macs = total_macs
    out.total_parameters = total_params
    out.weight_bytes = sum(l.weight_bytes for l in layers)
    out.peak_activation_bytes = max(l.activation_bytes for l in layers)
    return out


def optimization_curve(prof: ModelProfile, platform: str,
                       prune_rates=(0.0, 0.2, 0.4, 0.6, 0.8),
                       quantize: bool = True, host_ms: float | None = None) -> list:
    """Latency/energy vs prune rate (optionally quantized) for one platform."""
    rows = []
    for rate in prune_rates:
        p = prune_profile(prof, rate)
        if quantize:
            p = quantize_profile(p)
        rep = profile_report(p, platforms=(platform,), quantize=quantize, host_ms=host_ms)
        pick = next((r for r in rep if r["quantized"] == "int8"), rep[0])
        rows.append({"prune_rate": rate, **pick})
    return rows
