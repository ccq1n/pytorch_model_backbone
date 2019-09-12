''' SqueezeNet in PyTorch.

ICLR 2017

Reference:
[1] Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
    SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel, stride, pad=0, bias=False):
        super(Conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Fire(nn.Module):
    nums = 1
    def __init__(self, inp, mip, oup):
        super(Fire, self).__init__()
        self.layer1 = Conv_bn_relu(inp, mip, kernel=1, stride=1)
        self.layer2 = Conv_bn_relu(mip, oup, kernel=1, stride=1)
        self.layer3 = Conv_bn_relu(mip, oup, kernel=3, stride=1, pad=1)

    def forward(self, x):
        x = self.layer1(x)
        out1 = self.layer2(x)
        out2 = self.layer3(x)
        out = torch.cat([out1, out2], 1)
        out = F.relu(out, inplace=True)
        return out

class SqueezeNet(nn.Module):

    def __init__(self, block=Fire):
        super(SqueezeNet, self).__init__()
        self.conv1 = Conv_bn_relu(3, 96, kernel=7, stride=2, pad=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = block(96, 16, 64)
        self.fire3 = block(128, 16, 64)
        self.fire4 = block(128, 32, 128)
        self.layer1 = self._make_layer1()
        self.fire5 = block(256, 32, 128)
        self.fire6 = block(256, 48, 192)
        self.fire7 = block(384, 48, 192)
        self.fire8 = block(384, 64, 256)
        self.layer2 = self._make_layer2()
        self.fire9 = block(512, 64, 256)
        self.layer3 = self._make_layer3()
        self.conv2 = Conv_bn_relu(512, 1000, kernel=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=13)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.fire2(out)
        out = self.fire3(out)
        out = self.fire4(out)
        out = self.maxpool(out)
        out = self.fire5(out)
        out = self.fire6(out)
        out = self.fire7(out)
        out = self.fire8(out)
        out = self.maxpool(out)
        out = self.fire9(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        return out

def SqueezeNet_Simple():


def SqueezeNet_Complex():

def test():
    net = SqueezeNet()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

#test()
