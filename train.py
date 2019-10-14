'''Train ImageNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
resume = False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
    cfg.traindir, transforms.Compose([
        transforms.RandomResizedCrop(cfg.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cfg.valdir, transforms.Compose([
        transforms.Resize(int(cfg.input_size/0.875)),
        transforms.CenterCrop(cfg.input_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.n_worker, pin_memory=True)

print('==> Building model..')
net = ResNet50()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Lrate: %.5f | Loss: %.5f | Acc: %.5f%% (%d/%d)' % (cfg.lr, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def val(epoch):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    print("Correct Accuracy: %.5f | Best Accuracy: %.5f" % (acc, best_acc))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+2):
    train(epoch)
    print('Valing...')
    val(epoch)