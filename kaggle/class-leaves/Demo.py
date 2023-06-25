# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm

base_path = r'C:\Users\25351\ML-data\classify-leaves'
# 查看labels文件
labels_dataframe = pd.read_csv(base_path+'/dataset/train.csv')
# print(labels_dataframe)

# 把label文件排个序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# print(leaves_labels)
# print(n_classes)


# 把label转成对应的数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))

# print(class_to_num)

# 再转换回来，方便最后预测的时候使用
num_to_class = {v: k for k, v in class_to_num.items()}




train_path = base_path+'/dataset/train.csv'
test_path = base_path+'/dataset/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = base_path+'/dataset/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    # 是否打乱数据
    shuffle=False,

)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=16,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False
)


# 给大家展示一下数据长啥样
def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)

    return image


# fig = plt.figure(figsize=(20, 12))
# columns = 4
# rows = 2
#
# dataiter = iter(val_loader)
# inputs, classes = next(dataiter)
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     ax.set_title(num_to_class[int(classes[idx])])
#     plt.imshow(im_convert(inputs[idx]))
# plt.show()


# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
print(device)


# from efficientnet_pytorch import EfficientNet

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


# resnet34模型
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model_ft, feature_extract)
    # 改写全连接层的输出
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    # model = EfficientNet.from_name('efficientnet-b3')
    # model.load_state_dict(torch.load('./adv-efficientnet-b3-cdd7c0f4.pth'))
    # fc_features = model._fc.in_features
    # model._fc = nn.Linear(fc_features, num_classes)

    return model_ft


learning_rate = 2e-5
weight_decay = 1e-3
num_epoch = 18
model_path = 'pre_res_model.ckpt'

# Initialize a model, and put it on the device specified.
model = res_model(176)
model = model.to(device)
model.device = device
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# The number of training epochs.
n_epochs = num_epoch

best_acc = 0.0
for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()
    # These are used to record information in training.
    train_loss = []
    train_accs = []
    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs)
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):
        imgs, labels = batch
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))
