pandas



# PytorchDataset

Pytorch的dataset类是一个抽象类，继承dataset，需要实现它的__getitem__()方法和__len__()方法，下图是Pytorch官方文档中关于dataset类的说明。

实例代码

```python
class CashDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        纸币分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform
 
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     
 
        if self.transform is not None:
            img = self.transform(img)   
 
        return img, label
 
    def __len__(self):
        return len(self.data_info)
 
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
 
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))
 
        return data_info
```

上面代码中的静态方法get_img_info(data_dir)就是用来构建数据列表的，它返回数据列表data_info，data_info中的元素由元组(图像路径，图像标签)构成。

在__getitem__(self, index)方法中，通过data_info中存储的文件路径去读取图像数据，最后返回索引下标为index的图像数据和标签。这里返回哪些数据主要是由训练代码中需要哪些数据来决定。也就是说，我们根据训练代码需要什么数据来重写__getitem__(self, index)方法并返回相应的数据。

最后还要重写__len__(self)方法。实现__len__(self)方法比较简单，只需一行代码，也就是返回数据列表的的长度，即数据集的样本数量。

下面对构建CashDataset类做个小结，主要步骤如下：

1) 确定训练代码需要哪些数据；

2) 重写__getitem__(self, index)方法，根据index返回训练代码所需的数据；

3) 编写静态方法，构建并返回数据列表data_info；

4) 重写__len__(self)方法，返回数据列表长度；





# 分析 loss 和 val_loss (test_loss) 变化情况

通常回调显示的 loss 有很多种，如一个总 total_loss 多个子 sub_loss 。但本文主要分析最基础的训练情况（只有一个训练 loss，和一个验证 loss）。下文用 loss 代表训练集的损失值（墨守成规不写成 train_loss）；val_loss 代表验证集的损失值（也写成 test_loss）。

一般训练规律：

| loss     | val_loss   | 网络情况                                                     |
| -------- | ---------- | ------------------------------------------------------------ |
| 下降↓    | 下降↓      | 网络训练正常，最理想情况情况。                               |
| 下降↓    | 稳定/上升↑ | 网络过拟合。解决办法：①降低网络性能：在数据集没问题的前提下，向网络某些层的位置添加 Dropout 层（通常会选择较深的层，如一共 100 层，选择在 75 层；或者选择特征最多的层，如 Unet 的最底层等等）；或者逐渐减少网络的深度（靠经验删除部分模块）。<br/>②修改数据集：数据集有明显错误则需要重做，问题小可尝试混洗数据集并重新分配，通常开源数据集不容易出现这种情况。 |
| 稳定     | 下降↓      | 数据集有严重问题，建议重新选择。一般不会出现这种情况。       |
| 快速稳定 | 快速稳定   | 如果数据集规模不小的情况下，代表学习过程遇到瓶颈，需要减小学习率（自适应动量优化器小范围修改的效果不明显）。其次考虑修改 batchsize 大小。如果数据集很规模很小的话代表训练稳定。 |
| 上升↑    | 上升↑      | 可能是网络结构设计问题、训练超参数设置不当、数据集需要清洗等问题。<br/>这种情况属于训练过程中最差情况，得一个一个排除问题。 |

注意：上面提到的“下降”、“稳定”和“上升”是指整体训练趋势。