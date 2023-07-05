import torch
from torchvision import datasets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import os
# 自己的模块
from utils import train
from utils import plot_history
from Model.VisionTransformer import vit_cifar_patch4_32


transform = transforms.Compose([
    #     transforms.CenterCrop(224),
    transforms.RandomCrop(32, padding=4),  # 数据增广
    transforms.RandomHorizontalFlip(),  # 数据增广
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
epoch = 20
Batch_Size = 64
dataset_root = r'/data/lihan/cifar10'
train_set = datasets.CIFAR10(root=dataset_root, train=True, download=False, transform=transform)
test_set = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=Batch_Size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=Batch_Size, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = vit_cifar_patch4_32()

# optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94, verbose=True, patience=1,
                                                 min_lr=0.000001)  # 动态更新学习率
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.5)

if not os.path.exists('MyTransformer/model'):
    os.makedirs('MyTransformer/model')
else:
    print('文件已存在')
save_path = './model/vit_cifar.pth'

Acc, Loss, Lr = train(net, train_loader, test_loader, epoch,
                      optimizer, criterion, scheduler, save_path, verbose=True)
plot_history(epoch, Acc, Loss, Lr)
