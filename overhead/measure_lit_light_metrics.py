# -*- coding: utf-8 -*-
"""Measure ECE / inference latency / peak memory for lightweight literature models.

Fixed test set per dataset (single representative direction):
  A: UCI Drift, source=batch1 -> target=batch2 (1to2)
  B: CDSL, source=master -> target=slave1
  C: GSA, target conc=50ppb, source=100+200 (LOCO)
One seed (7) per model per dataset, base (source-only) training, EPOCHS=50.
Records macro_f1, accuracy, ECE, params, peak_mem_MB, inference_ms.

Output: main_results/table_46_lit_metrics_light.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '6'
import sys, math, time, traceback, argparse, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.models as tvm

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
sys.path.insert(0, str(REPO / 'experiments'))
sys.path.insert(0, str(REPO / 'overhead'))

from revision_core import (StandardScaler, stratified_parts, check_disjoint,
                           classification_metrics, ece)
from run_revision import load_drift
from run_revision import shuffled_pool
from run_cdsl_full import load_cdsl

OUT = PROJECT / 'main_results'
SEED = 7
EPOCHS = 50
CLASS_WEIGHTS = {1: 1., 2: 1., 3: 1., 4: 1.5, 5: 1., 6: 2.}

# ---------------- model builders (reuse from probe_lit_a_all) ----------------
def _conv1d_like(c, in_ch):
    return nn.Conv1d(in_ch, c.out_channels,
                     kernel_size=c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size,
                     stride=c.stride[0] if isinstance(c.stride, tuple) else c.stride,
                     padding=c.padding[0] if isinstance(c.padding, tuple) else c.padding,
                     dilation=c.dilation[0] if isinstance(c.dilation, tuple) else c.dilation,
                     groups=c.groups, bias=c.bias is not None)

def _pool1d_like(p):
    return nn.MaxPool1d(p.kernel_size[0] if isinstance(p.kernel_size, tuple) else p.kernel_size,
                        p.stride[0] if isinstance(p.stride, tuple) else p.stride,
                        p.padding[0] if isinstance(p.padding, tuple) else p.padding,
                        dilation=p.dilation[0] if isinstance(p.dilation, tuple) else p.dilation,
                        ceil_mode=p.ceil_mode)

def _avgpool1d_like(p):
    return nn.AvgPool1d(p.kernel_size[0] if isinstance(p.kernel_size, tuple) else p.kernel_size,
                        p.stride[0] if isinstance(p.stride, tuple) else p.stride,
                        p.padding[0] if isinstance(p.padding, tuple) else p.padding,
                        ceil_mode=p.ceil_mode)

def convert_2d_to_1d(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _conv1d_like(child, child.in_channels))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.BatchNorm1d(child.num_features, child.eps,
                                                child.momentum, child.affine,
                                                child.track_running_stats))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, _pool1d_like(child))
        elif isinstance(child, nn.AvgPool2d):
            setattr(module, name, _avgpool1d_like(child))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool1d(1))
        elif isinstance(child, nn.Linear):
            continue
        else:
            convert_2d_to_1d(child)
    return module

def _replace_last_linear(seq, num_classes):
    idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    last = idxs[-1]
    prev = idxs[-2] if len(idxs) > 1 else None
    in_feat = seq[prev].out_features if prev is not None else seq[last].in_features
    seq[last] = nn.Linear(in_feat, num_classes)

def build_lenet_flat(num_classes):
    class LeNet1DFlat(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 6, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(6, 16, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool1d(2)
            self.fc1 = nn.LazyLinear(120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        def forward(self, x):
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.flatten(1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    return LeNet1DFlat()

def build_sniff_multinose(num_classes):
    class SniffMultinose1D(nn.Module):
        def __init__(self, n_groups=4):
            super().__init__()
            self.n_groups = n_groups
            self.mlps = None
            self.num_classes = num_classes
        def _ensure(self, seg):
            if self.mlps is None or self.mlps[0][0].in_features != seg:
                self.mlps = nn.ModuleList(
                    [nn.Sequential(nn.Linear(seg, 32), nn.ReLU()) for _ in range(self.n_groups)])
                self.head = nn.Linear(self.n_groups * 32, self.num_classes)
        def forward(self, x):
            B, _, L = x.shape
            seg = L // self.n_groups
            self._ensure(seg)
            outs = []
            for i in range(self.n_groups):
                chunk = x[:, :, i * seg:(i + 1) * seg]
                pooled = chunk.mean(dim=1).squeeze(1)
                outs.append(self.mlps[i](pooled))
            return self.head(torch.cat(outs, dim=1))
    return SniffMultinose1D()

def build_sniff_convnet(num_classes):
    class SniffConvNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, num_classes)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    return SniffConvNet1D()

def build_sniff_resnet(num_classes):
    class ResBlock1D(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
            self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        def forward(self, x):
            r = torch.relu(self.c1(x))
            r = torch.relu(self.c2(r))
            return torch.relu(r + self.skip(x))
    class SniffResNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = ResBlock1D(1, 32)
            self.b2 = ResBlock1D(32, 64)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, num_classes)
        def forward(self, x):
            x = self.b1(x)
            x = self.b2(x)
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    return SniffResNet1D()

def build_gasnet(num_classes):
    class GasNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [1, 16, 16, 32, 32, 64, 64]
            self.blocks = nn.ModuleList()
            for i in range(6):
                b = nn.Sequential(
                    nn.Conv1d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU(),
                    nn.Conv1d(ch[i + 1], ch[i + 1], 3, padding=1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU())
                self.blocks.append(b)
                if i < 5:
                    self.blocks.append(nn.MaxPool1d(2))
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(ch[-1], num_classes)
        def forward(self, x):
            for m in self.blocks:
                x = m(x)
            x = self.gap(x).flatten(1)
            return self.fc(x)
    return GasNet1D()

def build_lstm(num_classes):
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x.transpose(1, 2))
            return self.fc(out[:, -1, :])
    return LSTMModel()

BUILDERS = {
    'LeNet-5': build_lenet_flat,
    'Sniff Multinose': build_sniff_multinose,
    'Sniff ConvNet': build_sniff_convnet,
    'Sniff ResNet': build_sniff_resnet,
    'GasNet': build_gasnet,
    'LSTM': build_lstm,
}
MODEL_ORDER = ['Sniff ConvNet', 'Sniff ResNet', 'GasNet', 'LSTM', 'LeNet-5', 'Sniff Multinose']


def train_base(model, xs, ys, seed, classes, steps):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    classes_arr = np.asarray(classes)
    lut = {int(c): i for i, c in enumerate(classes_arr)}
    ys = np.array([lut[int(v)] for v in ys])
    weights = np.array([CLASS_WEIGHTS.get(int(c), 1.) for c in classes_arr], np.float32)
    wt = torch.from_numpy(weights)
    with torch.no_grad():
        model(torch.zeros(1, 1, xs.shape[1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch = 128
    steps = steps or math.ceil((len(xs) + len(classes) * 60) / batch)
    xt = torch.from_numpy(np.asarray(xs, np.float32).reshape(len(xs), 1, -1))
    yt = torch.from_numpy(ys).long()
    model.train()
    for epoch in range(EPOCHS):
        order = np.resize(rng.permutation(len(xt)), steps * batch)
        for i in range(steps):
            ids = order[i * batch:(i + 1) * batch]
            x = xt[ids]; y = yt[ids]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y, weight=wt)
            loss.backward(); optimizer.step()
    return model


def predict(model, x, classes, measure_latency=True):
    model.eval()
    xx = torch.from_numpy(np.asarray(x, np.float32).reshape(len(x), 1, -1))
    # peak memory during one forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if measure_latency:
        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            # warmup
            _ = model(xx[:16])
            t0 = time.perf_counter()
            _ = model(xx)
            t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0 / len(x)  # per-sample
    with torch.no_grad():
        probs = torch.softmax(model(xx), 1).numpy()
    return np.asarray(classes)[probs.argmax(1)], probs, latency_ms


def peak_memory_mb(model, x):
    """Approximate CPU peak allocator memory via torch's cached/allocated (CPU has no true peak; use tracemalloc)."""
    import tracemalloc
    xx = torch.from_numpy(np.asarray(x, np.float32).reshape(len(x), 1, -1))
    tracemalloc.start()
    with torch.no_grad():
        model(xx[:64])
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def run_dataset(dataset, model_name, builder, x, y, source, target, classes, direction):
    pool, test = stratified_parts(target, y, [.2, .8], SEED)
    check_disjoint(source=source, test=test)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    model = builder(len(classes))
    t0 = time.time()
    model = train_base(model, xs, y[source], SEED, classes, steps)
    train_s = time.time() - t0
    p, pr, latency_ms = predict(model, xt, classes)
    m = classification_metrics(y[test], p, pr, classes)
    ece_val = ece(pr, y[test], classes)
    peak = peak_memory_mb(model, xt)
    n_params = sum(p_.numel() for p_ in model.parameters())
    del model; gc.collect()
    return dict(dataset=dataset, model=model_name, method='base', direction=direction, seed=SEED,
                macro_f1=m['macro_f1'], accuracy=m['accuracy'], ece=ece_val,
                params=n_params, params_M=n_params / 1e6,
                peak_mem_MB=peak, inference_ms=latency_ms, training_s=train_s,
                n_source=len(source), n_test=len(test))


def main():
    out_path = OUT / 'table_46_lit_metrics_light.csv'
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev['dataset'], prev['model']))
        print(f'resume: {len(prev)} rows, done: {sorted(done)}', flush=True)
    rows = []
    def flush_save(new_row=None):
        if new_row is not None:
            rows.append(new_row)
        new = pd.DataFrame(rows)
        if out_path.exists():
            prev = pd.read_csv(out_path)
            combined = pd.concat([prev, new], ignore_index=True)
        else:
            combined = new
        combined.to_csv(out_path, index=False, float_format='%.6f')
    # A
    x, y, b = load_drift(str(PROJECT))
    classes = np.arange(1, 7)
    for name in MODEL_ORDER:
        if ('A', name) in done:
            print(f'A skip {name}', flush=True)
            continue
        source = np.flatnonzero(b == 1)
        target = np.flatnonzero(b == 2)
        flush_save(run_dataset('A', name, BUILDERS[name], x, y, source, target, classes, '1to2'))
        print(f'A {name} done + saved', flush=True)
    # B
    xb, yb, src_b, tgt_b = load_cdsl('master_slave1')
    classes_b = np.arange(1, 7)
    for name in MODEL_ORDER:
        if ('B', name) in done:
            print(f'B skip {name}', flush=True)
            continue
        source = src_b
        target = tgt_b
        flush_save(run_dataset('B', name, BUILDERS[name], xb, yb, source, target, classes_b, 'master_slave1'))
        print(f'B {name} done + saved', flush=True)
    # C: GSA LOCO, target=50ppb, source=100+200
    from run_gsa_loco import load_gsa
    xc, yc, conc = load_gsa(str(PROJECT))
    yc = yc + 1  # 1..6
    classes_c = np.arange(1, 7)
    for name in MODEL_ORDER:
        if ('C', name) in done:
            print(f'C skip {name}', flush=True)
            continue
        source = np.flatnonzero(conc != 50)
        target = np.flatnonzero(conc == 50)
        flush_save(run_dataset('C', name, BUILDERS[name], xc, yc, source, target, classes_c, '50ppb'))
        print(f'C {name} done + saved', flush=True)
    print('saved ->', out_path, '| rows:', len(pd.read_csv(out_path)), flush=True)
    pd.set_option('display.width', 240)
    print(combined[['dataset', 'model', 'macro_f1', 'accuracy', 'ece', 'params_M',
                    'peak_mem_MB', 'inference_ms', 'training_s']].to_string(index=False))


if __name__ == '__main__':
    main()
