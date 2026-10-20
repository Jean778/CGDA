"""Per-layer analytical cost model: MACs, parameter/activation memory,
latency and energy across the three target platform classes.

Model description
-----------------
The CGDA backbone is an MLP with widths [128, 128, 64] plus a linear head
(6 classes). For a generic Linear layer ``in->out``:

    MACs         = in * out
    parameters   = in * out + out
    FLOPs        = 2 * MACs        (mul + add; reported as 1 MAC = 2 FLOPs)
    weight bytes = parameters * bytes_per_value
    activation   = out * bytes_per_value

Latency is MAC-bound for these tiny fully-connected layers:

    latency_ms = sum_layers( MACs_l / (mac_rate * efficiency) ) + fixed_overhead

Energy follows power x time. All platform coefficients are public-spec
estimates with a documented source class; the global efficiency coefficient
for the host anchor is calibrated against a real measurement (see fidelity.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BYTES = {"fp32": 4, "int8": 1}

# ---------------------------------------------------------------------------
# Platform profiles (public specifications; efficiency coefficients are
# analytical estimates, not measurements)
# ---------------------------------------------------------------------------
PLATFORMS = {
    # STM32H7-class MCU (Cortex-M7 @480 MHz, single-cycle FPU FMA,
    # CMSIS-NN Q7 kernels). STM32H743: 1 MB RAM, 2 MB flash, ~461 DMIPS.
    "stm32h7": {
        "class": "mcu",
        "cpu_mhz": 480.0,
        "macs_per_cycle_fp32": 1.0,   # single-cycle FMA on the M7 FPU
        "int8_speedup": 2.5,          # CMSIS-NN vs scalar FP32 loop (typical)
        "eff_fp32": 0.60,             # loop/load overhead factor
        "eff_int8": 0.55,
        "power_mw_active": 400.0,     # full-speed active draw (datasheet order)
        "power_mw_idle": 20.0,        # sleep-with-RAM-retention
        "ram_kb": 1056.0, "flash_kb": 2048.0,
        "source": "STM32H743 datasheet + CMSIS-NN public benchmarks",
    },
    # Raspberry Pi 4B (BCM2711, 4x Cortex-A72 @1.5 GHz, NEON 4xFP32 FMA).
    "rp4b": {
        "class": "sbc",
        "cpu_mhz": 1500.0,
        "cores": 4,
        "macs_per_cycle_fp32": 4.0,   # 128-bit NEON, 4x FP32 FMA per cycle
        "int8_speedup": 2.0,          # 8x INT8 per NEON cycle vs 4x FP32
        "eff_fp32": 0.25,             # single-thread real-world efficiency
        "eff_int8": 0.30,
        "power_mw_active": 4500.0,    # full board under load (USB-C 5V/3A)
        "power_mw_idle": 3000.0,
        "ram_kb": 4 * 1024 * 1024 / 8,  # not binding for 42 KB models
        "flash_kb": 0.0,
        "source": "BCM2711 specs; Pi 4B power measurements (public reviews)",
    },
    # Google Coral Edge TPU (USB Accelerator, INT8-only, 4 TOPS @2 W).
    "edge_tpu": {
        "class": "accelerator",
        "tops_int8": 4.0e12,          # peak INT8 MACs/s
        "power_mw_active": 2000.0,
        "power_mw_idle": 300.0,
        "fixed_latency_us": 400.0,    # driver + USB transfer + pipeline fill
        "int8_only": True,
        "source": "Coral official specs (4 TOPS, 2 W)",
    },
}

# MACs / s for the CPU-style platforms
def _mac_rate(p: dict, quantized: bool) -> float:
    if p["class"] == "accelerator":
        return p["tops_int8"]
    if quantized:
        # INT8 path: FP32 MAC rate scaled by the platform INT8 speedup
        # (CMSIS-NN Q7 kernels / NEON INT8 SIMD), with its own efficiency.
        return p["cpu_mhz"] * 1e6 * p["macs_per_cycle_fp32"] * p["int8_speedup"] * p["eff_int8"]
    return p["cpu_mhz"] * 1e6 * p["macs_per_cycle_fp32"] * p["eff_fp32"]


@dataclass
class LayerCost:
    name: str
    in_features: int
    out_features: int
    macs: int
    parameters: int
    weight_bytes: int
    activation_bytes: int


@dataclass
class ModelProfile:
    dim: int
    widths: list
    nclasses: int
    layers: list = field(default_factory=list)
    total_macs: int = 0
    total_parameters: int = 0
    weight_bytes: int = 0
    peak_activation_bytes: int = 0
    dtype: str = "fp32"

    def layer_summary(self):
        rows = []
        for l in self.layers:
            rows.append({
                "layer": l.name, "in": l.in_features, "out": l.out_features,
                "macs": l.macs, "parameters": l.parameters,
                "weight_bytes": l.weight_bytes, "activation_bytes": l.activation_bytes,
            })
        return rows


def model_from_linear(model, dim: Optional[int] = None, nclasses: Optional[int] = None,
                      dtype: str = "fp32") -> ModelProfile:
    """Extract a cost profile from a torch module with ``fe`` (Linear stack)
    and ``cls`` (final Linear), e.g. :class:`gasbench.methods.Network`.

    Fallback: build the standard CGDA MLP ``[dim, 128, 128, 64, nclasses]``
    when ``model`` is None.
    """
    if model is None:
        widths = [128, 128, 64]
        lin = [("fc1", dim, 128), ("fc2", 128, 128), ("fc3", 128, 64), ("head", 64, nclasses)]
    else:
        lin = []
        prev = None
        for i, m in enumerate(model.fe):
            if hasattr(m, "in_features"):
                lin.append((f"fc{i}", m.in_features, m.out_features))
        lin.append(("head", model.cls.in_features, model.cls.out_features))
        dim = dim or lin[0][1]
        nclasses = nclasses or lin[-1][2]
        widths = [o for _, _, o in lin[:-1]]

    b = BYTES[dtype]
    layers = []
    for name, i, o in lin:
        layers.append(LayerCost(
            name=name, in_features=i, out_features=o,
            macs=i * o, parameters=i * o + o,
            weight_bytes=(i * o + o) * b,
            activation_bytes=o * b,
        ))
    prof = ModelProfile(dim=dim, widths=widths, nclasses=nclasses, layers=layers, dtype=dtype)
    prof.total_macs = sum(l.macs for l in layers)
    prof.total_parameters = sum(l.parameters for l in layers)
    prof.weight_bytes = sum(l.weight_bytes for l in layers)
    prof.peak_activation_bytes = max(l.activation_bytes for l in layers)
    return prof


def platform_latency_energy(prof: ModelProfile, platform: str,
                            quantized: bool = False, host_ms: Optional[float] = None) -> dict:
    """Latency (ms) and energy (mJ) for one forward pass on `platform`.

    Parameters
    ----------
    prof : ModelProfile
    platform : str
        Key of :data:`PLATFORMS`.
    quantized : bool
        INT8 profile (also affects activation traffic).
    host_ms : float, optional
        Measured x86 end-to-end latency (ms) for the same profile; used as an
        anchor when available (fidelity bridge).
    """
    p = PLATFORMS[platform]
    if p["class"] == "accelerator":
        if quantized or p.get("int8_only"):
            mac_rate = p["tops_int8"]
            compute_ms = prof.total_macs / mac_rate * 1e3
            mem_ms = (prof.weight_bytes + prof.peak_activation_bytes) / (400e6) * 1e3  # USB 3.0 ~400 MB/s
            latency_ms = compute_ms + mem_ms + p["fixed_latency_us"] / 1e3
            energy_mj = p["power_mw_active"] * latency_ms / 1e3
            return {"latency_ms": latency_ms, "energy_mj": energy_mj,
                    "compute_ms": compute_ms, "mem_ms": mem_ms, "fixed_ms": p["fixed_latency_us"] / 1e3}
        raise ValueError("Edge TPU is INT8-only; pass quantized=True")

    rate = _mac_rate(p, quantized)
    compute_ms = prof.total_macs / rate * 1e3
    # Activation traffic is tiny for MLPs; add a fixed 5% pipeline overhead.
    latency_ms = compute_ms * 1.05
    energy_mj = p["power_mw_active"] * latency_ms / 1e3
    out = {"latency_ms": latency_ms, "energy_mj": energy_mj, "compute_ms": compute_ms}
    if host_ms is not None:
        # Calibrated: scale host-measured latency by the platform MAC-rate
        # ratio. The host rate is *measured-derived* (MACs / measured ms),
        # not an assumed specification.
        host_rate = prof.total_macs / (host_ms * 1e-3)
        calib_ms = host_ms * (host_rate / rate)
        out["calibrated_latency_ms"] = calib_ms
        out["calibrated_energy_mj"] = p["power_mw_active"] * calib_ms / 1e3
    return out


def profile_report(prof: ModelProfile, platforms=("stm32h7", "rp4b", "edge_tpu"),
                   quantize: bool = False, host_ms: Optional[float] = None) -> list:
    """Return one row per (platform, quantized) for the summary table."""
    rows = []
    for plat in platforms:
        if PLATFORMS[plat]["class"] == "accelerator":
            qs = [True]  # Edge TPU is INT8-only; label it as such
        else:
            qs = [False, True] if quantize else [False]
        for q in qs:
            r = platform_latency_energy(prof, plat, quantized=q, host_ms=host_ms)
            rows.append({
                "platform": plat, "quantized": "int8" if q else "fp32",
                "latency_ms": round(r["latency_ms"], 4),
                "energy_mj": round(r["energy_mj"], 6),
                "calibrated_latency_ms": round(r.get("calibrated_latency_ms", float("nan")), 4),
                "calibrated_energy_mj": round(r.get("calibrated_energy_mj", float("nan")), 6),
            })
    return rows
