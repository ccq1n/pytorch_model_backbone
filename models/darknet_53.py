'''DarkNet-53 in PyTorch.

YOLOv3

Reference:
[1] Joseph Redmon, Ali Farhadi
    YOLOv3: An Incremental Improvement
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1):
        super(Conv_bn_relu, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.convs(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inp):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv_bn_relu(inp, inp//2, kernel_size=1, stride=1, pad=0)
        self.conv2 = Conv_bn_relu(inp//2, inp, kernel_size=3, stride=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = F.relu(out)
        return out

class DarkNet_53(nn.Module):
    def __init__(self, num_blocks=[1,2,8,8,4], num_classes=1000):
        super(DarkNet_53, self).__init__()
        self.in_planes = 32

        self.conv1 = Conv_bn_relu(3, 32, kernel_size=3, stride=1)
        self.conv2 = Conv_bn_relu(32, 64, kernel_size=3, stride=2)
        self.resblock1 = self._make_layer(BasicBlock, 64, num_blocks[0])
        self.conv3 = Conv_bn_relu(64, 128, kernel_size=3, stride=2)
        self.resblock2 = self._make_layer(BasicBlock, 128, num_blocks[1])
        self.conv4 = Conv_bn_relu(128, 256, kernel_size=3, stride=2)
        self.resblock3 = self._make_layer(BasicBlock, 256, num_blocks[2])
        self.conv5 = Conv_bn_relu(256, 512, kernel_size=3, stride=2)
        self.resblock4 = self._make_layer(BasicBlock, 512, num_blocks[3])
        self.conv6 = Conv_bn_relu(512, 1024, kernel_size=3, stride=2)
        self.resblock5 = self._make_layer(BasicBlock, 1024, num_blocks[4])
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.resblock1(out)
        out = self.conv3(out)
        out = self.resblock2(out)
        out = self.conv4(out)
        out = self.resblock3(out)
        out = self.conv5(out)
        out = self.resblock4(out)
        out = self.conv6(out)
        out = self.resblock5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = DarkNet_53()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()