import torch
from torchinfo import summary

from Model.VisionTransformer import vit_cifar_patch4_32

net = vit_cifar_patch4_32()


def print_model_info(net):
    # (1,3,32,32) batch size  通道数  图片宽高
    summary(net, (1, 3, 32, 32))
    # print(net)


def test1():
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    print_model_info(net)
    # test1()
