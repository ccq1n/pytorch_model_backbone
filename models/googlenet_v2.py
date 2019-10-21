'''GoogleNet Inception-V2 in PyTorch.
2015

Reference:
[1] Sergey Ioffe, Christian Szegedy
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y = torch.cat([y1, y2, y3, y4], 1)
        return y

class Inception_S2(nn.Module):
    def __init__(self, in_planes, n3x3red, n3x3, n5x5red, n5x5):
        super(Inception_S2, self).__init__()

        # 1x1 conv -> 3x3 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool branch
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y = torch.cat([y1, y2, y3], 1)
        return y

class GoogleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNetV2, self).__init__()
        self.pre_layers_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pre_layers_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.a3 = Inception(192, 64, 64, 64, 64, 96, 32)
        self.b3 = Inception(256, 64, 64, 96, 64, 96, 64)
        self.c3 = Inception_S2(320, 128, 160, 64, 96)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(576, 224, 64, 96, 96, 128, 128)
        self.b4 = Inception(576, 192, 96, 128, 96, 128, 128)
        self.c4 = Inception(576, 160, 128, 160, 128, 160, 96)
        self.d4 = Inception(576, 96, 128, 192, 160, 192, 96)
        self.e4 = Inception_S2(576, 128, 192, 192, 256)

        self.a5 = Inception(1024, 352, 192, 320, 160, 224, 128)
        self.b5 = Inception(1024, 352, 192, 320, 192, 224, 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers_1(x)
        x = self.maxpool(x)
        x = self.pre_layers_2(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.c3(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def test():
     x = torch.randn(1,3,224,224)
     net = GoogleNetV2()
     y = net(x)
     print(y.size())

# test()