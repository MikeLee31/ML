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

epochs = 10
batch_size = 16
net = vit_cifar_patch4_32()
# 优化器
optimizer = optim.Adam(net.parameters())
# 损失函数
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94, verbose=True, patience=1,
                                                     min_lr=0.000001)  # 动态更新学习率
def main():
    dataset_root = r'/data/lihan/cifar10'

    # 分布式所需代码
    gpus=[0,1,2,3]
    net = nn.DataParallel(net.cuda(), device_ids=gpus, output_device=gpus[0])

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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    else:
        print('文件已存在')
    save_path = './checkpoint/vit_cifar.pth'

    Acc, Loss, Lr = train(net, train_loader, test_loader, epochs,
                          optimizer, criterion, scheduler, save_path, verbose=True)
    plot_history(epochs, Acc, Loss, Lr)


if __name__ == '__main__':
    main()
