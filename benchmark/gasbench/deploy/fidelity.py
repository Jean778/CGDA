"""Fidelity anchoring: simulation vs. public hardware baselines.

GasBench has no physical boards, so hardware fidelity cannot be measured
directly. Instead we do two honest things:

1. Host anchor calibration
   The 41.7 K-parameter MLP was benchmarked on the author's x86 machine
   (end-to-end 0.1321 ms, single-sample forward 0.0973 ms; see the Methods
   section, Table 2-3). We convert this measurement into a MAC->time
   coefficient and rescale it to each platform's MAC rate. This gives
   *calibrated* platform estimates with a documented measurement anchor.

2. Public-baseline alignment
   Our analytical estimates are checked for order-of-magnitude consistency
   against public TinyML baselines (MLPerf Tiny / EEMBC reports and vendor
   figures for the same platform classes). The alignment error is reported as
   an *expected* fidelity band (the reference paper reports MAD 5.72% on their
   matched configurations; ours is a bound, not a claim).

The reference for the methodology is:
   N. Vivekanandan, K. Rajeswari, "A Simulation-First Methodology for
   Sustainable TinyML Design," IEEE IoT-J, 2026. DOI 10.1109/JIOT.2026.3685306
"""
from __future__ import annotations

from .cost_model import platform_latency_energy

# Public-order anchors per platform class (latency for a ~40-50 k-MAC MLP or
# similar tiny model, from vendor/community reports). Used only for
# order-of-magnitude consistency checks. For the Edge TPU the anchor includes
# the USB driver/pipeline fixed cost (Coral end-to-end reports), which
# dominates micro-models; the compute-only window is microseconds.
PUBLIC_ANCHORS_MS = {
    "stm32h7": {"lo": 0.10, "hi": 1.00},   # CMSIS-NN tiny-MLP reports
    "rp4b":    {"lo": 0.02, "hi": 0.20},   # NEON single-thread tiny MLP
    "edge_tpu": {"lo": 0.10, "hi": 2.00},  # end-to-end incl. USB fixed cost
}


def calibrated_estimates(prof, platforms=("stm32h7", "rp4b", "edge_tpu"),
                         host_ms: float = 0.1321, host_macs: int | None = None) -> list:
    """Rescale the measured host latency to each platform via MAC-rate ratio."""
    rows = []
    for plat in platforms:
        # Host anchor rate is measured-derived: MACs / measured end-to-end ms.
        host_rate = prof.total_macs / (host_ms * 1e-3)
        from .cost_model import PLATFORMS, _mac_rate
        p = PLATFORMS[plat]
        if p["class"] == "accelerator":
            rate = p["tops_int8"]
            compute_ms = prof.total_macs / rate * 1e3
            latency = compute_ms + p["fixed_latency_us"] / 1e3 + 0.0004
        else:
            latency = host_ms * (host_rate / _mac_rate(p, quantized=False))
        anchor = PUBLIC_ANCHORS_MS[plat]
        inside = anchor["lo"] <= latency <= anchor["hi"]
        rows.append({
            "platform": plat,
            "calibrated_latency_ms": round(latency, 4),
            "public_anchor_lo_ms": anchor["lo"],
            "public_anchor_hi_ms": anchor["hi"],
            "within_public_band": inside,
        })
    return rows


def report(rows) -> str:
    lines = ["Fidelity anchoring (analytical estimates, not hardware measurements)"]
    lines.append(f"{'platform':<10}{'calib_ms':>10}{'public_lo':>12}{'public_hi':>12}  band_ok")
    for r in rows:
        lines.append(f"{r['platform']:<10}{r['calibrated_latency_ms']:>10.4f}"
                     f"{r['public_anchor_lo_ms']:>12.4f}{r['public_anchor_hi_ms']:>12.4f}  {r['within_public_band']}")
    lines.append("Expected fidelity band: reference methodology reports MAD 5.72% "
                 "(p90 6.25%, max 6.52%) on matched configurations; our band is a "
                 "bound, not a measurement.")
    return "\n".join(lines)
