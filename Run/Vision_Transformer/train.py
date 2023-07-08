import argparse

import torch
from torchvision import datasets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import os
import torch.distributed as dist
# 自己的模块
from utils import train
from utils import plot_history
from VisionTransformerModel import vit_cifar_patch4_32


def main():
    # 参数列表
    parser = argparse.ArgumentParser()
    # 分布式所需要代码
    parser.add_argument('--data', metavar='DIR', nargs='?', default='/data/lihan/cifar10',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    net = vit_cifar_patch4_32()

    net = net.to()

    # 优化器
    optimizer = optim.Adam(net.parameters())
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94, verbose=True, patience=1,
                                                     min_lr=0.000001)  # 动态更新学习率
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.5)

    transform = transforms.Compose([
        #     transforms.CenterCrop(224),
        transforms.RandomCrop(32, padding=4),  # 数据增广
        transforms.RandomHorizontalFlip(),  # 数据增广
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    else:
        print('文件已存在')
    save_path = './checkpoint/vit_cifar.pth'

    Acc, Loss, Lr = train(net, train_loader, test_loader, args.epochs,
                          optimizer, criterion, scheduler, save_path, verbose=True)
    plot_history(args.epochs, Acc, Loss, Lr)


if __name__ == '__main__':
    main()
