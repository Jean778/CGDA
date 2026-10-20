# -*- coding: utf-8 -*-
"""Driver: heavy models run per-model across A, B, C in one pass.
Order: for each heavy model in [AlexNet, GoogLeNet, ResNet34, VGG13, VGG19]:
    run on A -> run on B -> run on C (each saves incrementally).
After all 5 models are done on all 3 datasets, exit.
Interruption-safe: re-running resumes from where it left off.
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os, sys, subprocess, time
from pathlib import Path

PY = r'E:\anaconda3\python.exe'
CODE = REPO / 'code'
OUT = REPO / 'main_results'

A_SCRIPT = CODE / 'probe_lit_a_all.py'
BC_HEAVY = CODE / 'probe_lit_bc_heavy.py'
A_HEAVY = ['AlexNet', 'GoogLeNet', 'ResNet34', 'VGG13', 'VGG19']

DATASET_FILES = {
    'A': OUT / 'table_46_lit_a_1toK_all.csv',
    'B': OUT / 'table_46_lit_b_forward_all.csv',
    'C': OUT / 'table_46_lit_c_loco_all.csv',
}


def done_models(csv_path):
    if not csv_path.exists():
        return set()
    import pandas as pd
    return set(pd.read_csv(csv_path)['model'].unique())


def run(cmd):
    print(f'[{time.strftime("%H:%M:%S")}] RUN: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'!!! FAILED rc={r.returncode}: {" ".join(cmd)}', flush=True)
        return False
    return True


def main():
    for m in A_HEAVY:
        # A
        a_done = done_models(DATASET_FILES['A'])
        if m in a_done:
            print(f'A [{m}] already done, skip', flush=True)
        else:
            if not run([PY, str(A_SCRIPT), '--models', m]):
                sys.exit(1)
        # B
        b_done = done_models(DATASET_FILES['B'])
        if m in b_done:
            print(f'B [{m}] already done, skip', flush=True)
        else:
            if not run([PY, str(BC_HEAVY), '--only', 'B', '--models', m]):
                sys.exit(1)
        # C
        c_done = done_models(DATASET_FILES['C'])
        if m in c_done:
            print(f'C [{m}] already done, skip', flush=True)
        else:
            if not run([PY, str(BC_HEAVY), '--only', 'C', '--models', m]):
                sys.exit(1)
        print(f'*** MODEL {m} FINISHED on A+B+C ***', flush=True)

    print('ALL HEAVY MODELS DONE on A, B, C', flush=True)


if __name__ == '__main__':
    main()
