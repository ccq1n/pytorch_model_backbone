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
    def __init__(self, inp, oup, kernel=1, stride=1, pad=0, bias=False):
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
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = Conv_bn_relu(3, 96, kernel=7, stride=2, pad=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = Fire(96, 16, 64)
        self.fire3 = Fire(128, 16, 64)
        self.fire4 = Fire(128, 32, 128)
        self.fire5 = Fire(256, 32, 128)
        self.fire6 = Fire(256, 48, 192)
        self.fire7 = Fire(384, 48, 192)
        self.fire8 = Fire(384, 64, 256)
        self.fire9 = Fire(512, 64, 256)
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

class SqueezeNet_Simple(nn.Module):
    '''
    Base on SqueezeNet, adding bypass connections around Fire modules 3,5,7 and 9
    output of Fire2 + output of Fire3, where the + operator is elementwise addition.
    '''
    def __init__(self):
        super(SqueezeNet_Simple, self).__init__()
        self.conv1 = Conv_bn_relu(3, 96, kernel=7, stride=2, pad=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = Fire(96, 16, 64)
        self.fire3 = Fire(128, 16, 64)
        self.fire4 = Fire(128, 32, 128)
        self.fire5 = Fire(256, 32, 128)
        self.fire6 = Fire(256, 48, 192)
        self.fire7 = Fire(384, 48, 192)
        self.fire8 = Fire(384, 64, 256)
        self.fire9 = Fire(512, 64, 256)
        self.conv2 = Conv_bn_relu(512, 1000, kernel=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=13)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.fire2(out)
        out_fire3_beg = out
        out = self.fire3(out)
        out_fire3_end = out
        out = self.relu(out_fire3_end + out_fire3_beg)
        out = self.fire4(out)
        out = self.maxpool(out)
        out_fire5_beg = out
        out = self.fire5(out)
        out_fire5_end = out
        out = self.relu(out_fire5_beg + out_fire5_end)
        out = self.fire6(out)
        out_fire7_beg = out
        out = self.fire7(out)
        out_fire7_end = out
        out = self.relu(out_fire7_beg + out_fire7_end)
        out = self.fire8(out)
        out = self.maxpool(out)
        out_fire9_beg = out
        out = self.fire9(out)
        out_fire9_end = out
        out = self.relu(out_fire9_beg + out_fire9_end)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        return out

class SqueezeNet_Complex(nn.Module):
    def __init__(self):
        super(SqueezeNet_Complex, self).__init__()
        self.conv1 = Conv_bn_relu(3, 96, kernel=7, stride=2, pad=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = Fire(96, 16, 64)
        self.conv_fire2 = Conv_bn_relu(96, 128, kernel=1, stride=1)
        self.fire3 = Fire(128, 16, 64)
        self.fire4 = Fire(128, 32, 128)
        self.conv_fire4 = Conv_bn_relu(128, 256, kernel=1, stride=1)
        self.fire5 = Fire(256, 32, 128)
        self.fire6 = Fire(256, 48, 192)
        self.conv_fire6 = Conv_bn_relu(256, 384, kernel=1, stride=1)
        self.fire7 = Fire(384, 48, 192)
        self.fire8 = Fire(384, 64, 256)
        self.conv_fire8 = Conv_bn_relu(384, 512, kernel=1, stride=1)
        self.fire9 = Fire(512, 64, 256)
        self.conv2 = Conv_bn_relu(512, 1000, kernel=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=13)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out_2 = self.conv_fire2(out)
        out = self.fire2(out)
        out = self.relu(out + out_2)
        out_fire3_beg = out
        out = self.fire3(out)
        out_fire3_end = out
        out = self.relu(out_fire3_end + out_fire3_beg)
        out_4 = self.conv_fire4(out)
        out = self.fire4(out)
        out = self.relu(out + out_4)
        out = self.maxpool(out)
        out_fire5_beg = out
        out = self.fire5(out)
        out_fire5_end = out
        out = self.relu(out_fire5_beg + out_fire5_end)
        out_6 = self.conv_fire6(out)
        out = self.fire6(out)
        out = self.relu(out + out_6)
        out_fire7_beg = out
        out = self.fire7(out)
        out_fire7_end = out
        out = self.relu(out_fire7_beg + out_fire7_end)
        out_8 = self.conv_fire8(out)
        out = self.fire8(out)
        out = self.relu(out + out_8)
        out = self.maxpool(out)
        out_fire9_beg = out
        out = self.fire9(out)
        out_fire9_end = out
        out = self.relu(out_fire9_beg + out_fire9_end)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        return out

def test():
    # net = SqueezeNet()
    # net = SqueezeNet_Simple()
    net = SqueezeNet_Complex()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()
