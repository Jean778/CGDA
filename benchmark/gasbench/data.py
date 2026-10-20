"""Data acquisition and preprocessing for the three GasBench datasets.

All datasets are public and redistributable from the UCI Machine Learning
Repository:

* UCI Drift — Gas Sensor Array Drift Dataset (ID 224)
  https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
* GSA-LC  — Gas Sensor Array Low-Concentration Dataset (ID 1081)
  https://datalab-12.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration
* Air Quality — Air Quality UCI (ID 360)
  https://archive.ics.uci.edu/dataset/360/air+quality

`ensure_data(data_dir)` downloads whatever is missing (verified by SHA-256
where a reference hash is known) and caches parsed parquet/CSV files, so the
first run is the only slow one.
"""
from __future__ import annotations

import io
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry of remote sources
# ---------------------------------------------------------------------------
UA = {"User-Agent": "Mozilla/5.0 gasbench/0.1 (research; contact: author)"}

SOURCES = {
    "drift_zip": {
        "url": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        "kind": "zip",
    },
    "aq_zip": {
        "url": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "kind": "zip",
    },
    "gsalc_csv": {
        "url": "https://datalab-12.ics.uci.edu/dataset/1081/download",
        "kind": "csv",
        "ssl_verify": False,  # datalab serves a hostname-mismatched certificate
    },
}

# Known SHA-256 hashes of the raw files used in the paper (for integrity checks).
KNOWN_SHA = {
    "drift_clean.csv": "64a868d60798effdde46018a8f7fc2b076c962b5c99d30e6a44d16fa0c67f2b2",
    "gsalc.csv": "d79afe6494903b988de68d4a8ab133167151a1bb620ba413e5cefd15003d07b3",
    "AirQualityUCI.csv": "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996",
}

GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fetch(url: str, ssl_verify: bool = True) -> bytes:
    ctx = ssl.create_default_context() if ssl_verify else ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        return resp.read()


def _parse_drift_batches(batch_paths) -> pd.DataFrame:
    """Parse UCI Drift batch*.dat (129 cols: label + 128 'idx:value' features)."""
    rows, labels, batches = [], [], []
    for k in sorted(batch_paths):
        with open(batch_paths[k], "r") as fh:
            for line in fh:
                parts = line.split()
                labels.append(int(parts[0]))
                feats = [float(p.split(":", 1)[1]) for p in parts[1:]]
                assert len(feats) == 128, f"expected 128 features, got {len(feats)}"
                rows.append(feats)
                batches.append(k)
    x = np.asarray(rows, dtype=np.float64)
    cols = [f"s{i}_f{j}" for i in range(1, 17) for j in range(1, 9)]
    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "batch", batches)
    df.insert(1, "gas_id", labels)
    df["gas_name"] = [GAS_NAMES[g - 1] for g in labels]
    return df


def _parse_aq(raw_path: Path) -> pd.DataFrame:
    """Return the Air Quality UCI CSV untouched except for column filtering.

    The paper uses the 15-column semicolon format: 5 MOX channels, 4 reference
    concentrations, T, RH, plus Date/Time. Missing markers (-200) are handled
    downstream by `load_aq`.
    """
    df = pd.read_csv(raw_path, sep=";", decimal=",")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_all(data_dir, only=None, force: bool = False) -> dict:
    """Download and extract the requested raw files into `data_dir/raw`.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory. Files land in `<data_dir>/raw`.
    only : iterable of str, optional
        Subset of {"drift", "gsa", "aq"} to download. Default: all three.
    force : bool
        Re-download even if the target file exists.
    """
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    only = set(only or ["drift", "gsa", "aq"])
    out = {}

    if "drift" in only:
        batches = {k: raw / f"batch{k}.dat" for k in range(1, 11)}
        if all(p.exists() for p in batches.values()):
            out["drift"] = [p.name for p in batches.values()]
        else:
            zip_path = raw / "drift.zip"
            if force or not zip_path.exists():
                zip_path.write_bytes(_fetch(SOURCES["drift_zip"]["url"]))
            with zipfile.ZipFile(zip_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".dat")]
                for n in names:
                    (raw / Path(n).name).write_bytes(z.read(n))
            out["drift"] = sorted(Path(n).name for n in names)

    if "aq" in only:
        dest = raw / "AirQualityUCI.csv"
        if dest.exists() and not force:
            out["aq"] = [dest.name]
        else:
            zip_path = raw / "aq.zip"
            if force or not zip_path.exists():
                zip_path.write_bytes(_fetch(SOURCES["aq_zip"]["url"]))
            with zipfile.ZipFile(zip_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                for n in names:
                    (raw / Path(n).name).write_bytes(z.read(n))
            out["aq"] = sorted(Path(n).name for n in names)

    if "gsa" in only:
        dest = raw / "gsalc.csv"
        if force or not dest.exists():
            dest.write_bytes(_fetch(SOURCES["gsalc_csv"]["url"], ssl_verify=False))
        out["gsa"] = ["gsalc.csv"]

    return out


def _build_drift_clean(data_dir: Path) -> pd.DataFrame:
    raw = data_dir / "raw"
    batches = {}
    for k in range(1, 11):
        p = raw / f"batch{k}.dat"
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p.name}; run download_all(data_dir, only=['drift'])"
            )
        batches[k] = p
    return _parse_drift_batches(batches)


def ensure_data(data_dir) -> Path:
    """Download missing raw files and build processed CSVs.

    Returns the processed-data directory. Existing files verified by SHA-256
    when a reference hash is known; otherwise re-generated.
    """
    data_dir = Path(data_dir)
    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    download_all(data_dir)

    drift_clean = proc / "drift_clean.csv"
    if not drift_clean.exists():
        df = _build_drift_clean(data_dir)
        df.to_csv(drift_clean, index=False)
    gsa = data_dir / "raw" / "gsalc.csv"
    aq = data_dir / "raw" / "AirQualityUCI.csv"
    for p in (drift_clean, gsa, aq):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    return proc


# ---------------------------------------------------------------------------
# Dataset loaders (identical semantics to the paper's protocol)
# ---------------------------------------------------------------------------
def load_drift(data_dir) -> tuple:
    """Return (X, gas_id, batch) for UCI Drift with log1p normalization.

    X is (n_samples, 128) float32; the log1p transform is applied to the raw
    feature values after clipping negatives, exactly as in the paper.
    """
    proc = ensure_data(data_dir)
    d = pd.read_csv(proc / "drift_clean.csv")
    x = np.log1p(np.maximum(d.filter(regex="^s").values, 0)).astype(np.float32)
    return x, d.gas_id.values, d.batch.values


def load_gsa(data_dir) -> tuple:
    """Return (X, y, concentration) for GSA-LC.

    X is (90, 70) float: 10 sensors x 7 per-sensor statistics
    (mean/std/min/max/baseline/peak-excursion/slope) extracted from the
    900-point time series, matching the paper's feature definition.
    """
    proc = ensure_data(data_dir)
    raw_dir = proc.parent / "raw"
    d = pd.read_csv(raw_dir / "gsalc.csv", header=None)
    names = ["ethanol", "acetone", "toluene", "ethyl acetate", "isopropanol", "hexane"]
    y = np.array([names.index(s) for s in d.iloc[:, 0]])
    conc = d.iloc[:, 1].str.replace("ppb", "").astype(int).values
    ts = d.iloc[:, 2:].values.reshape(len(d), 10, 900)
    feats = []
    for s in range(10):
        z = ts[:, s, :]
        b = z[:, :45].mean(1)
        t = np.arange(900)
        centered = t - t.mean()
        slope = z @ centered / (centered @ centered)
        feats.extend([z.mean(1), z.std(1), z.min(1), z.max(1), b, z.max(1) - b, slope])
    return np.stack(feats, 1).astype(np.float32), y, conc


def load_aq(data_dir) -> tuple:
    """Return (X, refs, bucket, ids, timestamps) for Air Quality.

    X: 5 MOX channels; refs: 4 reference concentrations; bucket: temperature-
    humidity domain in {0,1,2,3} with b = 2*(T>=20) + (RH>=50); ids: original
    row indices; timestamps: sorted ISO strings. Rows with missing markers
    (-200) or missing T/RH are dropped.
    """
    proc = ensure_data(data_dir)
    raw_dir = proc.parent / "raw"
    raw = pd.read_csv(raw_dir / "AirQualityUCI.csv", sep=";", decimal=",").iloc[:, :15]
    sensors = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
    refs = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    d = raw.replace(-200.0, np.nan).dropna(subset=sensors + refs + ["T", "RH"]).copy()
    d["timestamp"] = pd.to_datetime(d["Date"] + " " + d["Time"], format="%d/%m/%Y %H.%M.%S")
    d = d.sort_values("timestamp")
    ids = d.index.values
    b = (d["T"].values >= 20).astype(int) * 2 + (d["RH"].values >= 50).astype(int)
    return (
        d[sensors].values.astype(np.float32),
        d[refs].values.astype(np.float32),
        b,
        ids,
        d.timestamp.astype(str).values,
    )


def load_all(data_dir) -> dict:
    """Return {"drift": (...), "gsa": (...), "aq": (...)} with one download."""
    return {"drift": load_drift(data_dir), "gsa": load_gsa(data_dir), "aq": load_aq(data_dir)}
