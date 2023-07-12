from torchinfo import summary
from FasterVitModel import faster_vit_2_224


def printModel(model,x):
    # x = (1,3,32,32)
    # (1,3,32,32) batch size  通道数  图片宽高
    summary(model, x)
    print(model)


if __name__ == '__main__':
    net = faster_vit_2_224()
    printModel(net,(1,3,224,224))
    