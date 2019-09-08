''' DarkNet-19 in Pytorch

YOLOv2

Reference:
[1] Joseph Redmon, Ali Farhadi
    YOLO9000: Better, Faster, Stronger
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
    def __init__(self, inp, num):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv_bn_relu(inp // 2, inp, kernel_size=3, stride=1)
        self.conv2 = Conv_bn_relu(inp, inp // 2, kernel_size=1, stride=1, pad=0)
        self.layers = self._make_layer(num)

    def _make_layer(self, num):
        layers = []
        for i in range(num):
            if(i%2==0):
                layers.append(self.conv1)
            else:
                layers.append(self.conv2)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class DarkNet_19(nn.Module):
    def __init__(self, growth_rate=2, num_blocks=[1,1,3,3,5,5], num_classes=1000):
        super(DarkNet_19, self).__init__()
        self.in_planes = 32

        self.layer1 = Conv_bn_relu(3, 32, kernel_size=3, stride=1)
        self.layer2 = BasicBlock(64, num_blocks[1])
        self.layer3 = BasicBlock(128, num_blocks[2])
        self.layer4 = BasicBlock(256, num_blocks[3])
        self.layer5 = BasicBlock(512, num_blocks[4])
        self.layer6 = BasicBlock(1024, num_blocks[5])
        self.linear = nn.Linear(1024, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.layer2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.layer3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.layer5(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = DarkNet_19()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()