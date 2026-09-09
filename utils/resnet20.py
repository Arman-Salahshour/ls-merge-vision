import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        '''wraps the parameter free shortcut so it can sit inside a module'''
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            '''option a shortcut, subsample then zero pad the new channels'''
            '''this adds no parameters so every kept conv stays at c_in in 16 32 64'''
            pad = planes // 4
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad), "constant", 0)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, num_blocks=(3, 3, 3), num_classes=100):
        super().__init__()
        '''channel width at the current depth, updated by _make_layer'''
        self.in_planes = 16
        '''stem, excluded from chunking because its filter size 27 does not divide 144'''
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        '''three stages at 16 32 64 channels, three blocks each'''
        self.layer1 = self._make_layer(16, num_blocks[0], 1)
        self.layer2 = self._make_layer(32, num_blocks[1], 2)
        self.layer3 = self._make_layer(64, num_blocks[2], 2)
        '''head stays at 100 outputs for every expert so all shapes match'''
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        '''only the first block of a stage downsamples'''
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        '''global average pool then classify'''
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20(num_classes=100):
    '''3 blocks per stage gives 6n+2 = 20 layers'''
    return ResNetCifar((3, 3, 3), num_classes)


if __name__ == "__main__":
    m = resnet20()
    '''sanity check the parameter count and one forward pass'''
    total = sum(p.numel() for p in m.parameters())
    print(f"resnet20 total trainable params: {total:,}")
    print(m(torch.randn(2, 3, 32, 32)).shape)
