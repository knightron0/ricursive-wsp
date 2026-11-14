"""ResNet-18 (CIFAR variant): the architecture whose weights the agent writes."""
import torch
from torch import nn
import torch.nn.functional as F

from wsp.data import MEAN, STD


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        widths, strides, cin = (64, 128, 256, 512), (1, 2, 2, 2), 64
        for i, (width, stride) in enumerate(zip(widths, strides), 1):
            setattr(self, f'layer{i}', nn.Sequential(BasicBlock(cin, width, stride), BasicBlock(width, width, 1)))
            cin = width
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


ARCHITECTURE = f"""ResNet-18, CIFAR variant (PyTorch module names = state_dict prefixes).
Input: uint8 RGB 32x32 -> /255 -> per-channel normalize, mean {MEAN}, std {STD}.
conv1: Conv2d(3->64, 3x3, stride 1, pad 1, no bias) -> bn1 -> ReLU          -> 64x32x32
layer1: 2 BasicBlocks, 64 ch, stride 1                                     -> 64x32x32
layer2: 2 BasicBlocks, 128 ch, first block stride 2                        -> 128x16x16
layer3: 2 BasicBlocks, 256 ch, first block stride 2                        -> 256x8x8
layer4: 2 BasicBlocks, 512 ch, first block stride 2                        -> 512x4x4
avgpool: global average -> 512;  fc: Linear(512 -> num_classes) -> logits
BasicBlock(x): out = ReLU(bn1(conv1(x))); out = bn2(conv2(out)); return ReLU(out + shortcut(x)).
conv1 is 3x3 with the block's stride, conv2 is 3x3 stride 1, all convs have no bias.
shortcut = identity, except layer{{2,3,4}}.0 where shortcut.0 is a 1x1 strided conv and shortcut.1 a BatchNorm.
BatchNorm (eval mode): y = (x - running_mean) / sqrt(running_var + 1e-5) * weight + bias.
State dict entries per BatchNorm: weight, bias, running_mean, running_var (float), num_batches_tracked (int64, unused).
Conv weights are [out, in, kh, kw]; fc.weight is [num_classes, 512]."""


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def state_shapes(num_classes):
    return {name: list(t.shape) for name, t in ResNet18(num_classes).state_dict().items()}


def initial_weights(num_classes, seed=0):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in ResNet18(num_classes).state_dict().items()}
