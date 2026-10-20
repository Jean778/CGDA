# CGDA — Centroid-Guided Data Augmentation for Sensor Drift Compensation

Code accompanying the paper *Class-Centroid-Guided Mixup for Domain Adaptation toward
Sensor Drift Compensation*.

CGDA is a calibration-guided domain-adaptation method whose adaptation mechanism is
data augmentation: with a fixed label budget, a few labelled target-domain calibration
samples per class are used (i) as supervised signal appended to the source training set
(`cal_sup`), and (ii) as class centroids in the standardized input space that anchor
constrained Mixup interpolations along the source-sample → target-centroid direction
(`cg_sup`). The same feature extractor supports closed-set recognition, open-set
rejection and environmental-event detection through cosine prototype distances and
threshold estimation.

## Repository layout

| Directory | Contents |
|---|---|
| `core/` | Shared primitives: model, training, data splits, metrics, ECE (`revision_core.py`) |
| `experiments/` | Experiment entry points (one script per protocol/dataset) |
| `data/` | Dataset download/preprocessing scripts (raw data is not distributed) |
| `tables/` | Aggregation and paper-table assembly scripts |
| `figures/` | Figure generation scripts (R, matching the paper figures) |
| `overhead/` | Computational-overhead measurement scripts (params/FLOPs/latency/peak memory) |
| `benchmark/` | `gasbench` package (MIT) used by the INT8 quantization study |

## Requirements

- Python ≥ 3.9
- R ≥ 4.0 (figures only) with `ggplot2`, `dplyr`, `tidyr`, `ggrepel`, `patchwork`

```bash
pip install numpy pandas scipy scikit-learn torch torchvision matplotlib thop psutil
pip install -e benchmark   # optional: only needed for experiments/run_deploy_quant.py
```

## Datasets

Raw data is **not** included. Four public benchmarks are used; expected locations and loaders:

| Dataset | Source | Path expected by loaders |
|---|---|---|
| UCI Gas Sensor Array Drift (A) | [UCI ML Repository #224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) | `data/processed/drift_clean.csv` — `data/download_drift.py` fetches the raw batches and preprocesses them |
| CDSL sensor-drift (B) | Zhang et al., *IEEE TIM* 66(7):1679–1692 (2017), "Odor Recognition in Multiple E-nose Systems via Cross-domain Discriminative Subspace Learning"; download [CDSL_paper_code.rar](https://www.leizhang.tk/resources/CDSL_paper_code.rar) | `data/CDSL/*.mat` → `data/CDSL_csv/*.csv` — `data/convert_cdsl_mat.py` converts the .mat arrays; source = master device, targets = slave1/slave2 (6 features, 6 gases) |
| Gas Sensor Array Low-concentration (C) | [UCI ML Repository #1081](https://archive.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration) | `data/raw/gsalc.csv` |
| UCI Air Quality (D) | [UCI ML Repository #360](https://archive.ics.uci.edu/dataset/360/air+quality) | `data/raw/AirQualityUCI.csv` |

## Running the experiments

Scripts resolve the repository root automatically from their own location. Add the
library directories to `PYTHONPATH` first:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD\core;$PWD\experiments;$PWD\data;$PWD\overhead"
# Linux / macOS
export PYTHONPATH="$PWD/core:$PWD/experiments:$PWD/data:$PWD/overhead"
```

Main entry points (each prints/CSVs its outputs under `results/` or `main_results/`):

- `experiments/run_revision.py` — main protocol suite (UCI forward/LOBO, GSA, AQ)
- `experiments/run_correct_protocols.py`, `run_lobo_target_first.py`, `run_forward_budget_scan.py` — Sec 4.2 protocols
- `experiments/run_ablation_supplement.py` — ablation methods (cg_mix / cg_shuffle / mean_shift) and feature metrics
- `experiments/run_osr_cosine.py` — open-set rejection (cosine distance)
- `experiments/run_aq_cosine.py` — environmental-event detection
- `experiments/run_gsa_loco.py` — GSA-LC leave-one-concentration
- `experiments/run_cdsl_full.py` — CDSL full suite (6 methods × 5 budgets × 2 directions × 5 seeds)
- `experiments/run_single_source.py` — UCI Drift single-source multi-target (source = batch k−1 → targets k..10)
- `experiments/run_heavy_models.py` — heavy literature models across A/B/C
- `tables/aggregate_*.py`, `build_*.py` — table assembly from `results/`
- `figures/plot_*.R` — figure generation (paper figures)

## Frozen configuration

Everything below is fixed in the code and must be preserved for reproduction:

- **Random seeds**: `SEEDS = [42, 2024, 7, 123, 2025]` (`core/revision_core.py`)
- **Determinism**: `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`,
  `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`
- **Protocol version**: `VERSION = "cgda_protocol_v2_20260909"`
- **Data splits**: per-class stratified 2:8 pool/test with ≥1 sample per class in the pool
  (`stratified_parts`); calibration = `β × per-class target size`, capped by pool availability
  (`nested_calibration`); source/calibration/test strictly disjoint (`check_disjoint`);
  `StandardScaler` fit on the source only
- **Network**: shared extractor = ReLU MLP 128→128→64 (small variant 32→16),
  linear classifier head; input dimension = dataset feature count
- **Training**: Adam lr=1e-3 (small 5e-3), batch=128, 50 epochs (AQ 80), class-weighted CE
- **Augmentation (cg_sup)**: 60 augmented samples per class per epoch (small 40),
  interpolation `(1−λ)·x_source + λ·centroid`, λ∼Beta(0.5, 0.5)
- **GSA-LC lightweight variant**: 2 calibration samples per class, LogisticRegression,
  centroid = 0.5×source + 0.5×calibration class mean, λ scaled by 0.6, 10 samples/class

## Citation

If you use this code or the CGDA method in your research, please cite our paper:

```bibtex
@article{peng2026cgda,
  title   = {Class-Centroid-Guided Mixup for Domain Adaptation toward Sensor Drift Compensation},
  author  = {Peng, Jingyi},
  year    = {2026},
  note    = {To appear}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).
`benchmark/gasbench` is also MIT.
