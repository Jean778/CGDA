"""Smoke tests: data loading, protocol planning, one tiny run, aggregation."""
import pytest

import gasbench as gb
from gasbench.protocols import task_list, execute_task, aggregate


def test_methods_known():
    for m in ["base", "cal_sup", "cg_mix", "cg_sup", "cg_shuffle", "mixup_std", "dann"]:
        assert m in gb.list_methods()


def test_task_plan_counts():
    tasks = task_list(seeds=[42], families=["closed", "osr", "gsa"])
    fams = {}
    for t in tasks:
        fams[t["family"]] = fams.get(t["family"], 0) + 1
    assert fams["closed"] == 10
    assert fams["osr"] == 3 * 10  # three unknown-split settings
    assert fams["gsa"] == 3


def test_stratified_parts_disjoint(tmp_path):
    import numpy as np
    from gasbench.utils import stratified_parts, check_disjoint
    y = np.repeat([1, 2, 3], 30)
    idx = np.arange(len(y))
    a, b, c = stratified_parts(idx, y, [0.2, 0.8], 42)
    check_disjoint(a=a, b=b, c=c)
    assert set(a) | set(b) | set(c) == set(idx)


def test_smoke_task_and_aggregate(tmp_path):
    import os
    data = str(tmp_path)
    out = str(tmp_path / "out")
    # Copy the tiny drift csv (no network in tests).
    import shutil
    from gasbench import data as gd
    src = os.environ.get("GASBENCH_TEST_DATA")
    if not src:
        pytest.skip("set GASBENCH_TEST_DATA to a directory with processed/drift_clean.csv")
    shutil.copytree(src, data, dirs_exist_ok=True)
    task = {"family": "closed", "setting": "lobo", "budget": 0.1, "seed": 42, "fold": 1}
    tid, status, secs = execute_task(task, data, out, epochs=2)
    assert status in ("complete", "cached")
    aggregate(out)
    import pandas as pd
    s = pd.read_csv(os.path.join(out, "summary_mean_sd.csv"))
    assert len(s) > 0
