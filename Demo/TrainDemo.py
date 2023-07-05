import torch
from d2l.torch import d2l
from torch import nn
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


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_model(model, n_epochs, train_loader, val_loader, lr, device, model_path, weight_dacay=5e-4, resume=False):
    if resume:
        print(f"Load Model From{model_path}")
        check_point = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(check_point)
    model.to(device)
    best_acc = 0
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_dacay)

    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        for batch in train_loader:
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            # 计算loss和精度
            acc = (Y_hat.argmax(dim=-1) == Y).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}", end='  ')

        model.eval()
        valid_loss = []
        valid_accs = []
        for batch in val_loader:
            X, Y = batch
            with torch.no_grad():
                Y_hat = model(X.to(device))
            loss = criterion(Y_hat, Y.to(device))
            acc = (Y_hat.argmax(dim=-1) == Y.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # 保存模型
        if train_acc > best_acc:
            best_acc = train_acc
        torch.save(model.state_dict(), model_path)
        if train_acc > 0.85:
            torch.save(model.state_dict(), './model85.ckpt')
            return
