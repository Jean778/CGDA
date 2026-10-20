"""Fleet-scale energy accounting.

Converts per-inference energy into annual kWh for an inspection-robot fleet
under a duty cycle, mirroring the sustainability metric of the reference
TinyML paper (32-259 kWh/year per 10k nodes there; ours differs because the
CGDA MLP is ~50x smaller than their 1D-CNN).
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_YEAR = 365.25 * 24 * 3600
J_PER_KWH = 3.6e6


def fleet_energy(inference_mj: float, wireless_mj: float, inferences_per_hour: float,
                 nodes: int, active_fraction: float = 1.0, baseline_mw: float = 0.0) -> dict:
    """Annual energy (kWh) for `nodes` nodes doing `inferences_per_hour`
    inferences each, every hour of the year.

    `active_fraction` discounts periods when the node is off (e.g. night
    shifts); both inference and wireless energy scale linearly with it.
    `baseline_mw` adds a system baseline (sensor sampling + MCU retention +
    radio standby), because for a 42 KB model the inference energy alone is
    negligible and would otherwise dominate the comparison misleadingly.
    """
    per_inference_j = (inference_mj + wireless_mj) * 1e-3
    inference_annual_j = per_inference_j * inferences_per_hour * 24 * 365.25 * nodes * active_fraction
    baseline_annual_j = baseline_mw * 1e-3 * SECONDS_PER_YEAR * nodes * active_fraction
    annual_j = inference_annual_j + baseline_annual_j
    return {
        "nodes": nodes,
        "inferences_per_hour": inferences_per_hour,
        "active_fraction": active_fraction,
        "per_inference_mj": inference_mj + wireless_mj,
        "baseline_mw": baseline_mw,
        "annual_kwh": annual_j / J_PER_KWH,
        "annual_kwh_per_node": annual_j / J_PER_KWH / nodes,
        "inference_kwh": inference_annual_j / J_PER_KWH,
        "baseline_kwh": baseline_annual_j / J_PER_KWH,
    }


def duty_cycle_matrix(inference_mj: float, wireless_mj: float, nodes: int,
                      freqs=(6, 60, 600), fractions=(0.3, 0.6, 1.0),
                      baseline_mw: float = 0.0) -> np.ndarray:
    """(freq x fraction) annual-kWh grid, useful for the deployment table."""
    out = np.zeros((len(freqs), len(fractions)))
    for i, f in enumerate(freqs):
        for j, a in enumerate(fractions):
            out[i, j] = fleet_energy(inference_mj, wireless_mj, f, nodes, a,
                                     baseline_mw=baseline_mw)["annual_kwh"]
    return out
