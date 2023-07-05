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
    parser = argparse.ArgumentParser()
    # 分布式所需要代码
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    print(args.local_rank)
    # 分布式所需要代码
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    dataset_root = r'/data/lihan/cifar10'
    device = 'cuda:0,1,2,3'
    net = vit_cifar_patch4_32()
    # 分布式所需代码
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

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

    train_set = datasets.CIFAR10(root=dataset_root, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 分布式所需代码,sampler=
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

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
