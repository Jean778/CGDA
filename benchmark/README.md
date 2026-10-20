# GasBench

**A reproducible benchmark harness for drift-aware gas-sensor recognition.**

GasBench packages the full evaluation stack behind a recent study on
*calibration-guided domain adaptation* (CGDA) for gas sensors into a single
command-line tool. It bundles three public datasets, six protocol families,
eight training mechanisms, and a standardized aggregation/reporting contract,
so that new drift-compensation methods can be benchmarked against published
numbers with a single command.

```
pip install -e .
python -m gasbench --data ./data --out ./results --families closed,osr --workers 4
```

## Why this harness

Existing gas-sensor drift studies are hard to compare: each uses a different
data split, a different evaluation protocol, and a different set of baselines.
GasBench fixes the *protocol*, not the models:

- **Explicit domain splits.** Leave-one-batch-out, chronological
  (past-to-future), and single-source-to-many-target splits are defined once
  and shared by all methods.
- **No label leakage.** Calibration samples are selected by class-stratified,
  seed-driven partitions; the calibration selection never inspects features or
  model outputs (`nested_calibration`), and disjointness is asserted
  (`check_disjoint`).
- **Matched compute.** Every neural method runs the same MLP backbone with the
  same number of optimizer updates, so accuracy differences reflect the
  training mechanism rather than capacity or training budget.
- **Verified reproducibility.** Task execution is cached and hashed; results
  directories record data SHA-256, library versions, seeds, and the protocol
  version.

## Installation

Python ≥ 3.9. Dependencies: `numpy`, `pandas`, `scikit-learn`, `torch` (CPU
build is sufficient).

```bash
pip install -e ./gasbench            # from this repository
# or, from anywhere:
# pip install git+https://github.com/your-org/gasbench.git
```

## Datasets

GasBench downloads all three public datasets on first run from the UCI Machine
Learning Repository (no manual step):

| Dataset | UCI ID | Content | Role in the paper |
|---|---|---|---|
| Gas Sensor Array Drift | 224 | 6 gases, 16 MOX sensors × 8 features, 10 temporal batches (13,910 rows) | drift classification |
| Gas Sensor Array Low-Concentration | 1081 | 6 gases, 10 sensors × 900-point series at 50/100/200 ppb (90 samples) | cross-concentration transfer |
| Air Quality | 360 | 5 MOX channels + 4 reference gases + T/RH, hourly (≈9,000 valid rows) | domain/event detection |

`python -m gasbench --data ./data --download-only` fetches and verifies them.

## Protocols

| Family | Task | Split | Calibration |
|---|---|---|---|
| `closed` | 6-class drift classification | leave-one-batch-out (LOBO) | β = 0.1 of target per class |
| `budget` | same, sensitivity to β | LOBO | β ∈ {0.05, 0.1, 0.2} |
| `forward` | chronological drift | train on past batches, test on the next | β = 0.1 |
| `singlesource` | cold-start drift | one source batch → every other batch | β = 0.1 |
| `osr` | open-set rejection | known classes train; held-out + unknown-class samples test | β = 0.1 + threshold pool |
| `gsa` | cross-concentration | train on 2 concentrations, test on the third (each direction) | 2 samples/class |
| `gsa_cv` | small-sample CV | 5-fold stratified CV | 2 samples/class |
| `aq` | domain & event | temperature/humidity blocks, chronological | ≤15 samples/domain + threshold pool |

Every task is run over the 5 seeds `[42, 2024, 7, 123, 2025]`; per-seed fold
means are then averaged and reported as `mean ± std` across seeds.

## Methods

| Method | Description |
|---|---|
| `base` | MLP trained on source labels only |
| `cal_sup` | MLP fine-tuned on source + a tiny calibration set |
| `cg_mix` | calibration labels + vanilla Mixup |
| `cg_sup` | **CGDA**: calibration labels + class-centroid-guided Mixup |
| `cg_shuffle` | negative control: centroids assigned to wrong classes |
| `mixup_std` | standard Mixup without calibration |
| `dann` | gradient-reversal domain adaptation (uses the calibration set as unlabeled target) |
| `logreg` / `sklearn_mlp` | classical baselines |
| `mean_shift` | input-space mean correction with a `base` model |

### Adding a method

```python
from gasbench.methods import register_method

@register_method("my_method")
def train_my_method(xs, ys, xc, yc, seed, classes, epochs, steps, small=False):
    model = ...
    return model, {"parameter_count": ..., "optimizer_steps": ...}
```

Registered methods are picked up automatically by every protocol; they must
return a module exposing `.fe` and `.cls` (used by `network_output`).

## Output contract

Every task directory `results/tasks/<family>_<setting>_.../` contains:

- `metrics.csv` — per-metric values (accuracy, macro-F1, ECE, per-class F1)
- `model_details.csv` — parameter count, optimizer steps, wall-clock seconds
- `splits.csv` / `preprocessing.csv` — exact index split and scaler statistics
- `predictions_<method>.csv` — per-sample predictions, probabilities, distances

Aggregates at the results root:

- `summary_mean_sd.csv` — seed-mean ± seed-SD per (family, method, metric)
- `summary_by_fold_mean_sd.csv` — fold-resolved summary
- `paired_differences_mean_sd.csv` — paired method deltas for significance
- `task_status.csv`, `data_manifest.csv`, `environment.csv`, `planned_tasks.csv`

## Deployment profiling (simulation-first)

`gasbench deploy` profiles the CGDA models on heterogeneous edge platforms
without any physical hardware, following the simulation-first methodology of
Vivekanandan & Rajeswari (IEEE IoT-J 2026, DOI 10.1109/JIOT.2026.3685306):

```bash
python -m gasbench deploy --quantize --prune 0.3 --prune-curve \
    --wireless ble,wifi,lora,nb_iot --cloud --baseline-mw 100 \
    --out results/deploy
```

What it computes:

- **Per-layer cost model** — MACs, parameters, weight/activation bytes for the
  41.7 K-parameter MLP; latency and energy from public platform profiles
  (STM32H7-class MCU, Raspberry Pi 4B, Edge TPU).
- **Hardware-aware optimization** — INT8 PTQ (4x memory compression, ~2-2.5x
  MCU speedup) and structured neuron pruning (latency/energy vs prune-rate
  curve).
- **Wireless system view** — BLE / Wi-Fi / LoRa / NB-IoT link delay and energy
  for edge (label-only) vs cloud (raw 128-feature) modes.
- **Fleet energy** — annual kWh for N nodes under duty cycles, including a
  system baseline (sensor + MCU + radio standby).
- **Fidelity anchoring** — host x86 measurement (0.1321 ms) rescaled to each
  platform's MAC rate and checked against public TinyML baseline bands.

> All platform numbers are analytical estimates from public specifications;
> GasBench has no physical boards. The only measurement is the x86 host anchor.
> Do not present the estimates as hardware measurements.

Outputs land in `results/deploy/` as `platform_profile.csv`, `prune_curve.csv`,
`system_level.csv`, `fleet_energy.csv`, `fidelity.csv`.

## Quick reproduction of the paper numbers

```bash
python -m gasbench --data ./data --out ./paper --families closed,budget,forward,singlesource,osr,gsa,aq --workers 8
```

On a 12-thread CPU this runs in a few hours (the full paper grid ≈ 1,700 tasks
at ≈10 s/task single-threaded; the closed family alone is ≈ 1.5 h with 8
workers).

## Citation

If you use GasBench in your research, please cite the companion paper (details
in `CITATION.cff`):

> J. Peng, "Calibration-Guided Domain Adaptation for Gas Sensor Drift
> Recognition: A Unified Framework for Closed-Set, Open-Set and Event
> Detection," *to appear*.

## License

MIT — see `LICENSE`. The bundled datasets are redistributed under the UCI
Machine Learning Repository terms; cite the original dataset papers when
publishing.

## Notes

- The `datalab-12.ics.uci.edu` mirror used for GSA-LC serves a
  hostname-mismatched TLS certificate; GasBench disables verification for that
  one request only. Audit `gasbench/data.py` before use in restricted networks.
- `torch` runs single-threaded inside workers for deterministic, reproducible
  training; increase `--workers` rather than BLAS threads to speed up.
