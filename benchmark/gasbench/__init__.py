"""GasBench: drift-aware gas sensor recognition benchmark.

A reproducible harness that bundles three public gas-sensor datasets
(UCI Drift, GSA-LC, Air Quality), six evaluation protocols (closed LOBO /
forward / single-source, open-set rejection, concentration drift,
anomaly detection) and a set of drift-compensation methods including the
calibration-guided domain adaptation (CGDA) framework.

Typical usage (command line):

    python -m gasbench --data ./data --out ./results --families closed,osr --smoke

Or as a library:

    from gasbench.data import load_all
    x, y, batch = load_all("./data")["drift"]
"""

__version__ = "0.2.0"

from .data import (
    load_drift,
    load_gsa,
    load_aq,
    load_all,
    download_all,
    ensure_data,
)
from .utils import check_disjoint, stratified_parts, nested_calibration, SEEDS
from .evaluate import (
    ece,
    classification_metrics,
    oscr,
    rejection_metrics,
    binary_metrics,
    quantile_threshold,
    seed_summary,
)
from .methods import (
    Network,
    train_network,
    network_output,
    distance_model,
    distances,
    list_methods,
    register_method,
)

__all__ = [
    "__version__",
    "load_drift",
    "load_gsa",
    "load_aq",
    "load_all",
    "download_all",
    "ensure_data",
    "check_disjoint",
    "stratified_parts",
    "nested_calibration",
    "SEEDS",
    "ece",
    "classification_metrics",
    "oscr",
    "rejection_metrics",
    "binary_metrics",
    "quantile_threshold",
    "seed_summary",
    "Network",
    "train_network",
    "network_output",
    "distance_model",
    "distances",
    "list_methods",
    "register_method",
]
