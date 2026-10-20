"""Deployment simulation for GasBench (v0.2).

Simulation-first profiling of the CGDA models on heterogeneous edge
platforms, following the methodology of Vivekanandan & Rajeswari (IEEE IoT-J,
2026, DOI 10.1109/JIOT.2026.3685306): per-layer analytical cost models,
hardware-aware INT8 quantization and structured pruning, wireless-link
integration, fleet-scale energy accounting, and fidelity anchoring against
public hardware baselines.

Honesty contract
----------------
GasBench has **no physical MCU / SBC / TPU hardware**. Every platform number
below is an analytical estimate built from public specifications (datasheet
figures, MLPerf Tiny / EEMBC reports, vendor throughput claims) with explicit
efficiency coefficients. The only measured anchor is x86 host inference
(0.13 ms end-to-end for the 41.7 K-parameter MLP, measured on the author's
machine), which is used to calibrate the global MAC->time coefficient. Do not
present these estimates as hardware measurements.
"""
