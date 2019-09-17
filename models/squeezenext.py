''' SqueezeNext in Pytorch

Reference:
[1] Amir Gholami, Kiseok Kwon, Bichen Wu, Zizheng Tai, Xiangyu Yue, Peter Jin, Sicheng Zhao, Kurt Keutzer
    SqueezeNext: Hardware-Aware Neural Network Design
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, pad=0):
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
    def __init__(self, inp, oup, res=False):
        super(BasicBlock, self).__init__()
        self.inp = inp
        self.oup = oup
        self.res = res
        self.layer1 = Conv_bn_relu(inp, oup//2)
        self.layer2 = Conv_bn_relu(oup//2, oup//4)
        self.layer3 = Conv_bn_relu(oup//4, oup//2)
        self.layer4 = Conv_bn_relu(oup//2, oup//2)
        self.layer5 = Conv_bn_relu(oup//2, oup)
        self.layer6 = Conv_bn_relu(inp, oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        if self.inp != self.oup or self.res:
            out = self.relu(self.layer6(x) + out)
        return out

class SqueezeNext(nn.Module):
    def __init__(self, num_block=[6,6,8,1], channel=[32,64,128,256], num_classes=1000):
        super(SqueezeNext, self).__init__()
        self.conv1 = Conv_bn_relu(3, 64, kernel_size=7, stride=2, pad=3)
        self.layer1 = self._make_layer(64, channel[0], num_block[0])
        self.layer2 = self._make_layer(channel[0], channel[1], num_block[1])
        self.layer3 = self._make_layer(channel[1], channel[2], num_block[2])
        self.layer4 = self._make_layer(channel[2], channel[3], num_block[3])
        self.conv2 = Conv_bn_relu(channel[3], 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, inp, oup, num_block):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(BasicBlock(inp, oup, res=True))
            else:
                layers.append(BasicBlock(oup, oup))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.maxpool(out)
        out = self.layer2(out)
        out = self.maxpool(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SqNxt_x1_23():
    return SqueezeNext([6,6,8,1], [32,64,128,256])

def SqNxt_x1_23v5():
    return SqueezeNext([2,4,14,1], [32,64,128,256])

def SqNxt_x2_23():
    return SqueezeNext([6,6,8,1], [64,128,256,512])

def SqNxt_x2_23v5():
    return SqueezeNext([2,4,14,1], [64,128,256,512])

def test():
    net = SqNxt_x2_23()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()