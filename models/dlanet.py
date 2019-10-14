''' DLANet in PyTorch.

cvpr2018

https://arxiv.org/abs/1707.06484

Reference:
[1] Fisher Yu Dequan Wang Evan Shelhamer Trevor Darrell
    Deep Layer Aggregation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inp, oup, stride=1, pad=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = oup // expansion
        self.conv1 = nn.Conv2d(inp, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, oup, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32
    def __init__(self, inp, oup, stride=1, pad=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = oup * cardinality // 32 # avoid 'oup' not enough to conv group

        self.conv1 = nn.Conv2d(inp, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride,
                               padding=pad, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, oup, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class Root(nn.Module):
    def __init__(self, inp, oup, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            inp, oup, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, inp, oup, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 pad=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * oup
        if level_root:
            root_dim += inp
        if levels == 1:
            self.tree1 = block(inp, oup, stride=stride, pad=pad)
            self.tree2 = block(oup, oup, stride=1, pad=pad)
        else:
            self.tree1 = Tree(levels-1, block, inp, oup,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              pad=pad, root_residual=root_residual)
            self.tree2 = Tree(levels-1, block, oup, oup,
                              root_dim=root_dim+oup,
                              root_kernel_size=root_kernel_size,
                              pad=pad, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, oup, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if inp != oup:
            self.project = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup)
            )
    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLANet(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False):
        super(DLANet, self).__init__()

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_layer(3, channels[0])
        self.level1 = self._make_layer(channels[0], channels[1])
        self.level2 = Tree(levels[2], block, channels[1], channels[2],
                           stride=2, level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3],
                           stride=2, level_root=True,
                           root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4],
                           stride=2, level_root=True,
                           root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5],
                           stride=1, level_root=True,
                           root_residual=residual_root)

        self.linear = nn.Linear(channels[5], num_classes)

    def _make_layer(self, inp, oup, stride=1, pad=1):
        modules = []
        modules.append(BasicBlock(inp, inp, stride=stride, pad=pad))
        modules.append(nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        ))
        # modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*modules)

    def forward(self, x):
        out = self.base_layer(x)
        out = self.level0(out)
        out = self.level1(out)
        out = self.level2(out)
        out = self.level3(out)
        out = self.level4(out)
        out = self.level5(out)
        return out

def DLANet34():
    return DLANet([1,1,1,2,2,1], [16,32,64,128,256,512], block=BasicBlock)

def DLANet46_C():
    return DLANet([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=Bottleneck)

def DLANet60():
    return DLANet([1, 1, 1, 2, 3, 1],[16, 32, 128, 256, 512, 1024], block=Bottleneck)

def DLANet102():
    return DLANet([1, 1, 1, 3, 4, 1],[16, 32, 128, 256, 512, 1024], block=Bottleneck)

def DLANet169():
    return DLANet([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck)

def DLANetX46_C():
    return DLANet([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX)

def DLANetX60_C():
    return DLANet([1, 1, 1, 2, 3, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX)

def DLANetX60():
    return DLANet([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX)

def DLANetX102():
    return DLANet([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX)

def test():
    net = DLANetX60_C()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()