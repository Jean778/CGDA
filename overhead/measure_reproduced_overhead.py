# -*- coding: utf-8 -*-
"""Reproduce open-source / well-defined models as 1D versions and measure
Params + MACs + FLOPs with thop at input dims 70 / 128 / 252 (single-variable:
same 1D input format, only architecture differs).
All models share a unified 6-class head; only input dimension varies.
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import torch
import torch.nn as nn
import torchvision.models as tvm
from thop import profile
import pandas as pd
from pathlib import Path

OUT = REPO / 'main_results'
DIMS = [70, 128, 252]
NUM_CLASSES = 6


# ---------------- 2D -> 1D conversion utilities ----------------
def _conv1d_like(c: nn.Conv2d, in_ch):
    return nn.Conv1d(in_ch, c.out_channels,
                     kernel_size=c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size,
                     stride=c.stride[0] if isinstance(c.stride, tuple) else c.stride,
                     padding=c.padding[0] if isinstance(c.padding, tuple) else c.padding,
                     dilation=c.dilation[0] if isinstance(c.dilation, tuple) else c.dilation,
                     groups=c.groups, bias=c.bias is not None)


def _pool1d_like(p: nn.MaxPool2d):
    return nn.MaxPool1d(p.kernel_size[0] if isinstance(p.kernel_size, tuple) else p.kernel_size,
                        p.stride[0] if isinstance(p.stride, tuple) else p.stride,
                        p.padding[0] if isinstance(p.padding, tuple) else p.padding,
                        dilation=p.dilation[0] if isinstance(p.dilation, tuple) else p.dilation,
                        ceil_mode=p.ceil_mode)


def _avgpool1d_like(p: nn.AvgPool2d):
    return nn.AvgPool1d(p.kernel_size[0] if isinstance(p.kernel_size, tuple) else p.kernel_size,
                        p.stride[0] if isinstance(p.stride, tuple) else p.stride,
                        p.padding[0] if isinstance(p.padding, tuple) else p.padding,
                        ceil_mode=p.ceil_mode)


def convert_2d_to_1d(module):
    """Recursively swap Conv2d/BatchNorm2d/Pool2d to 1d variants (in-place, returns new)."""
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


# ---------------- model builders ----------------
def _replace_last_linear(seq, num_classes):
    """Replace the last nn.Linear in a Sequential (skip Dropout/ReLU) with num_classes head."""
    idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    last = idxs[-1]
    prev = idxs[-2] if len(idxs) > 1 else None
    in_feat = seq[prev].out_features if prev is not None else seq[last].in_features
    seq[last] = nn.Linear(in_feat, num_classes)


def _replace_first_linear(seq, in_features):
    """Replace the first nn.Linear in a Sequential (skip Dropout/ReLU)."""
    idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    first = idxs[0]
    seq[first] = nn.Linear(in_features, seq[first].out_features)
    return seq[first]


def build_vgg(variant='vgg13'):
    net = tvm.vgg13(weights=None) if variant == 'vgg13' else tvm.vgg19(weights=None)
    convert_2d_to_1d(net)  # handles features + avgpool (AdaptiveAvgPool2d -> 1d)
    # first conv: input channel 3 -> 1 (1D feature vector)
    net.features[0] = _conv1d_like(net.features[0], 1)
    # VGG classifier[0] is a Linear directly
    net.classifier[0] = nn.Linear(512, net.classifier[0].out_features)
    _replace_last_linear(net.classifier, NUM_CLASSES)
    return net


def build_alexnet():
    net = tvm.alexnet(weights=None)
    convert_2d_to_1d(net)  # features + avgpool
    net.features[0] = _conv1d_like(net.features[0], 1)
    _replace_first_linear(net.classifier, 256)  # classifier[0] is Dropout; first Linear gets 256
    _replace_last_linear(net.classifier, NUM_CLASSES)
    return net


def build_googlenet():
    net = tvm.googlenet(weights=None, aux_logits=False)
    convert_2d_to_1d(net)
    # fix first conv input channel 3 -> 1 (BasicConv2d wrapper: replace inner conv)
    net.conv1.conv = _conv1d_like(net.conv1.conv, 1)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    return net


def build_resnet34():
    net = tvm.resnet34(weights=None)
    convert_2d_to_1d(net)
    net.conv1 = _conv1d_like(net.conv1, 1)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    return net


def build_lstm():
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1,
                                batch_first=True)
            self.fc = nn.Linear(64, NUM_CLASSES)

        def forward(self, x):
            # x: (B, 1, L) -> (B, L, 1)
            out, _ = self.lstm(x.transpose(1, 2))
            return self.fc(out[:, -1, :])

    return LSTMModel()


def build_lenet():
    class LeNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 6, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(6, 16, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(16, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, NUM_CLASSES)

        def forward(self, x):
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.mean(dim=2)  # global average over length
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    return LeNet1D()


def build_gasnet():
    """GasNet (Peng et al. 2018, Sensors 18(1):157): 6 conv blocks, each
    block = Conv1d+BN+ReLU+Conv1d+BN+ReLU; global average pool + FC head."""
    class GasNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [1, 16, 16, 32, 32, 64, 64]  # block input channels
            self.blocks = nn.ModuleList()
            for i in range(6):
                b = nn.Sequential(
                    nn.Conv1d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU(),
                    nn.Conv1d(ch[i + 1], ch[i + 1], 3, padding=1), nn.BatchNorm1d(ch[i + 1]), nn.ReLU(),
                )
                self.blocks.append(b)
                if i < 5:
                    self.blocks.append(nn.MaxPool1d(2))
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(ch[-1], NUM_CLASSES)

        def forward(self, x):
            for m in self.blocks:
                x = m(x)
            x = self.gap(x).flatten(1)
            return self.fc(x)

    return GasNet1D()


def build_sniff_convnet():
    """Sniff ConvNet (Serra et al. 2020, arXiv:2005.01611): 2 conv layers + 2 FC."""
    class SniffConvNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, NUM_CLASSES)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SniffConvNet1D()


def build_sniff_resnet():
    """Sniff ResNet (Serra et al. 2020): 2 residual blocks (2 Conv each) + 2 FC."""
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
            self.fc2 = nn.Linear(32, NUM_CLASSES)

        def forward(self, x):
            x = self.b1(x)
            x = self.b2(x)
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SniffResNet1D()


def build_sniff_multinose():
    """Sniff Multinose (Serra et al. 2020): per-column MLPs fused by another MLP.
    1D adaptation: split the length into 4 groups; each group's segment (seg dims)
    feeds a small MLP, outputs concatenated -> head. Seg length depends on input dim,
    so the sub-MLPs are built lazily on first forward (per input dim)."""
    class SniffMultinose1D(nn.Module):
        def __init__(self, n_groups=4):
            super().__init__()
            self.n_groups = n_groups
            self.mlps = None

        def _ensure(self, seg):
            if self.mlps is None or self.mlps[0][0].in_features != seg:
                self.mlps = nn.ModuleList(
                    [nn.Sequential(nn.Linear(seg, 32), nn.ReLU()) for _ in range(self.n_groups)])
                self.head = nn.Linear(self.n_groups * 32, NUM_CLASSES)

        def forward(self, x):
            B, _, L = x.shape
            seg = L // self.n_groups
            self._ensure(seg)
            outs = []
            for i in range(self.n_groups):
                chunk = x[:, :, i * seg:(i + 1) * seg]     # (B,1,seg)
                pooled = chunk.mean(dim=1).squeeze(1)      # (B,seg)
                outs.append(self.mlps[i](pooled))
            cat = torch.cat(outs, dim=1)
            return self.head(cat)

    return SniffMultinose1D()


# ---------------- measurement ----------------
def measure(model, dim):
    model.eval()
    x = torch.randn(1, 1, dim)
    with torch.no_grad():
        model(x)  # warm-up: triggers lazy module construction (e.g. Sniff Multinose)
    macs, params = profile(model, inputs=(x,), verbose=False)
    return params, macs


builders = {
    'VGG13': lambda: build_vgg('vgg13'),
    'VGG19': lambda: build_vgg('vgg19'),
    'AlexNet': build_alexnet,
    'GoogLeNet': build_googlenet,
    'ResNet34': build_resnet34,
    'LSTM': build_lstm,
    'LeNet-5': build_lenet,
    'GasNet': build_gasnet,
    'Sniff ConvNet': build_sniff_convnet,
    'Sniff ResNet': build_sniff_resnet,
    'Sniff Multinose': build_sniff_multinose,
}

rows = []
for name, builder in builders.items():
    model = builder()
    for dim in DIMS:
        params, macs = measure(model, dim)
        rows.append(dict(model=name, input_dim=dim, num_classes=NUM_CLASSES,
                         params=params, params_M=params / 1e6,
                         macs=macs, flops=macs * 2, flops_M=macs * 2 / 1e6))

t = pd.DataFrame(rows)
t.to_csv(OUT / 'table_46_reproduced_overhead.csv', index=False)
pd.set_option('display.width', 220)
print(t.pivot_table(index='model', columns='input_dim', values=['params_M', 'flops_M']).to_string(
    formatters={'params_M': '{:.4f}'.format, 'flops_M': '{:.4f}'.format}))
print()
print('saved -> main_results/table_46_reproduced_overhead.csv  ({} rows)'.format(len(t)))
