from torchinfo import summary

from MyTransformer.Vit.VisionTransformer import vit_cifar_patch4_32


net = vit_cifar_patch4_32()
summary(net,(1,3,32,32))