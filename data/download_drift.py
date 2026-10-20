# -*- coding: utf-8 -*-
"""
Download UCI Gas Sensor Array Drift Dataset (ID: 224).
Try the official zip archive; fall back to per-file download of batch1.dat..batch10.dat.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Project root = parent of this script's directory
BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-download"}

ZIP_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
OLD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/"
BATCH_NAMES = [f"batch{i}.dat" for i in range(1, 11)]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    # Plan 1: official zip archive
    try:
        print(f"[1/2] Trying official zip: {ZIP_URL}")
        data = fetch(ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            print("    zip contents:", names[:12], "..." if len(names) > 12 else "")
            for n in names:
                if n.lower().endswith((".dat", ".csv", ".txt")):
                    (RAW / Path(n).name).write_bytes(z.read(n))
        print("    zip extracted")
        return 0
    except Exception as e:
        print(f"    zip plan failed: {e}")

    # Plan 2: legacy per-file download
    print("[2/2] Falling back to per-file download")
    for name in BATCH_NAMES:
        url = OLD_URL + name
        try:
            (RAW / name).write_bytes(fetch(url))
            print(f"    {name} downloaded")
        except Exception as e:
            print(f"    {name} failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
