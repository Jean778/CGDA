# -*- coding: utf-8 -*-
"""Benchmark single-task runtime of heavy literature models on A 1->2 (CPU).
Only for timing; prints per-model seconds for one task."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '6'
import sys, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision.models as tvm

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
from revision_core import StandardScaler, stratified_parts
from run_revision import load_drift


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


def build_vgg(variant, nc=6):
    net = tvm.vgg13(weights=None) if variant == 'vgg13' else tvm.vgg19(weights=None)
    convert_2d_to_1d(net)
    net.features[0] = _conv1d_like(net.features[0], 1)
    net.classifier[0] = nn.Linear(512, net.classifier[0].out_features)
    _replace_last_linear(net.classifier, nc)
    return net


def build_alexnet(nc=6):
    net = tvm.alexnet(weights=None)
    convert_2d_to_1d(net)
    net.features[0] = _conv1d_like(net.features[0], 1)
    idxs = [i for i, m in enumerate(net.classifier) if isinstance(m, nn.Linear)]
    net.classifier[idxs[0]] = nn.Linear(256, net.classifier[idxs[0]].out_features)
    _replace_last_linear(net.classifier, nc)
    return net


def build_googlenet(nc=6):
    net = tvm.googlenet(weights=None, aux_logits=False)
    convert_2d_to_1d(net)
    net.conv1.conv = _conv1d_like(net.conv1.conv, 1)
    net.fc = nn.Linear(net.fc.in_features, nc)
    return net


def build_resnet34(nc=6):
    net = tvm.resnet34(weights=None)
    convert_2d_to_1d(net)
    net.conv1 = _conv1d_like(net.conv1, 1)
    net.fc = nn.Linear(net.fc.in_features, nc)
    return net


def build_lstm(nc=6):
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
            self.fc = nn.Linear(64, nc)
        def forward(self, x):
            out, _ = self.lstm(x.transpose(1, 2))
            return self.fc(out[:, -1, :])
    return LSTMModel()


def build_gasnet(nc=6):
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
            self.fc = nn.Linear(ch[-1], nc)
        def forward(self, x):
            for m in self.blocks:
                x = m(x)
            x = self.gap(x).flatten(1)
            return self.fc(x)
    return GasNet1D()


def build_sniff_convnet(nc=6):
    class SniffConvNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, nc)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    return SniffConvNet1D()


def build_sniff_resnet(nc=6):
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
            self.fc2 = nn.Linear(32, nc)
        def forward(self, x):
            x = self.b1(x)
            x = self.b2(x)
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    return SniffResNet1D()


BUILDERS = {
    'VGG13': lambda: build_vgg('vgg13'),
    'VGG19': lambda: build_vgg('vgg19'),
    'AlexNet': build_alexnet,
    'GoogLeNet': build_googlenet,
    'ResNet34': build_resnet34,
    'LSTM': build_lstm,
    'GasNet': build_gasnet,
    'Sniff ConvNet': build_sniff_convnet,
    'Sniff ResNet': build_sniff_resnet,
}


def run_base(model, xs, ys, seed, classes, steps, epochs=50):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    classes = np.asarray(classes); lut = {int(c): i for i, c in enumerate(classes)}
    ys = np.array([lut[int(v)] for v in ys])
    wt = torch.ones(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    xt = torch.from_numpy(np.asarray(xs, np.float32).reshape(len(xs), 1, -1))
    yt = torch.from_numpy(ys).long()
    batch = 128
    model.train()
    for epoch in range(epochs):
        order = np.resize(rng.permutation(len(xt)), steps * batch)
        for i in range(steps):
            ids = order[i * batch:(i + 1) * batch]
            loss = nn.functional.cross_entropy(model(xt[ids]), yt[ids], weight=wt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def main():
    x, y, b = load_drift(str(PROJECT))
    source = np.flatnonzero(b == 1)
    target = np.flatnonzero(b == 2)
    pool, test = stratified_parts(target, y, [.2, .8], 7)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    print(f'source n={len(source)} steps={steps}')

    torch.set_num_threads(6)
    for name, builder in BUILDERS.items():
        model = builder()
        t0 = time.time()
        run_base(model, xs, y[source], 7, np.arange(1, 7), steps, epochs=50)
        print(f'{name}: {time.time()-t0:.1f}s (1 task, A 1->2, seed7)', flush=True)


if __name__ == '__main__':
    main()
