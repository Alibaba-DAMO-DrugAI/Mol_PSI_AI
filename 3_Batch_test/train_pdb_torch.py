
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# from pytorch_fitmodule import FitModule
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# import revnet

# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

best_acc = 0

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def npy_loader(path):
    samples = torch.from_numpy(np.load(path))
    return samples

#当然出来的时候已经全都变成了tensor
class Trainset(Dataset):
    def __init__(self, X_path, y_train_path, loader=npy_loader):
        #定义好 image 的路径
        self.loader = npy_loader
        self.y_train = self.loader(y_train_path)
        self.X_train = self.loader(X_path)[:self.y_train.shape[0],]

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return len(self.X_train)    

class Testsetset(Dataset):
    def __init__(self, X_path, y_test_path, loader=npy_loader):
        #定义好 image 的路径
        self.loader = npy_loader
        self.y_test = self.loader(y_test_path)
        self.X_test = self.loader(X_path)[-self.y_test.shape[0]:,]

    def __getitem__(self, index):
        return self.X_test[index], self.y_test[index]

    def __len__(self):
        return len(self.X_test) 

def ResNet():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(36, 8, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2)
    )
 
    # 添加残差层
    # ...... #
    
    model.add_module("global_avg_pool", GlobalAvgPool2d())
    model.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear()))
    return model

class GlobalAvgPool2d(torch.nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
       return F.avg_pool2d(x, kernel_size=x.size()[2:])
 
class FlattenLayer(torch.nn.Module):
    # 用于全连接层
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    :param in_channels: 输入层通道数
    :param out_channels: 输出层通道数
    :param num_residuals: 残差层数
    :param first_block: 是否是第一个resnet块
    :return: 
    '''
    # 第一个模块的通道数同输入通道数一致。
    # 由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。
    # 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
    if first_block:
        # 第一个块 输入和输出的通道数需一致
        assert in_channels == out_channels
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=1))
        else:
            block.append(Residual(out_channels, out_channels))
    return torch.nn.Sequential(*block)

class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        '''
        :param in_channels: 输入的通道数
        :param out_channels: 输出的通道数
        :param use_1x1conv: 是否使用1*1卷积层 
        :param stride: 步长
        '''
        super(Residual, self).__init__()
        # kernel_size=3, padding=1, stride=1保证输入与输出宽高一致
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        # 批量归一化
        self.b1 = torch.nn.BatchNorm2d(out_channels)
        self.b2 = torch.nn.BatchNorm2d(out_channels)
 
    def forward(self, X):
        Y = F.relu(self.b1(self.conv1(X)))
        Y = self.b2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        # 输出和输入，此时Y = F(X) - X为残差层的输出，所以残差层实际拟合的是F(X)-X
        return F.relu(Y + X)


model = ResNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#第一行代码

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
scheduler = StepLR(optimizer, 30, gamma = 0.1)

trainset = Trainset("PDB_data/2016_norm_img/X_train_test.npy", "PDB_data/y_train_2016.npy")

testset = Trainset("PDB_data/2016_norm_img/X_train_test.npy", "PDB_data/y_test_2016.npy")

trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=500,
                                              shuffle=True, num_workers=4)

valloader = torch.utils.data.DataLoader(testset,
                                            batch_size=500,
                                            shuffle=False, num_workers=4)

for epoch in tqdm(range(100)):
    start = time.time()
    for X, y in trainloader:
        X = X.type(torch.FloatTensor).to(device)
        y = y.type(torch.FloatTensor).to(device)
 
        y_pre = model(X)
 
        l = loss(y_pre, y)
            # 梯度清零
        optimizer.zero_grad()
 
        l.backward()
        optimizer.step()
 
        train_loss_sum += l.cpu().item()
        train_rmse_sum += torch.sqrt(((y_pre-y)**2).sum()).cpu().item()
        n += y.shape[0]


valid_rmse = validation(model, test_batch)
print("validation rmse:", valid_rmse)
 

x = np.random.randn(1000, 8) * 1.1
print(predict(model, x, batch_size))



model = model.to(device)
    print("run in " , device)
    # 损失函数,MSE函数
    loss = RMSELoss()
 
    for epoch in range(num_epochs):
        train_loss_sum, n, batch_count = 0.0, 0, 0
        start = time.time()
 
        for X, y in train_batch:
            # 转置
            X = X.to(device)
            y = y.to(device)
 
            # 前向计算
            y_pre = model(X)
 
            l = loss(y_pre, y)
            # 梯度清零
            optimizer.zero_grad()
 
            l.backward()
            optimizer.step()
 
            train_loss_sum += l.cpu().item()
            train_rmse_sum += torch.sqrt(((y_pre-y)**2).sum()).cpu().item()
            n += y.shape[0]
            batch_count += 1
 
        test_rmse = evaluate_rmse(test_batch, model)
 
        print("epoch:%d, loss:%.4f, train_rmse:%.3f, test_rmse %.3f, cost: %.1f sec" %
              (epoch + 1, train_loss_sum / batch_count, train_rmse_sum / n, test_rmse, time.time() - start))


best_acc


if CUDA:
    model.cuda()

criterion = RMSELoss

optimizer = optim.SGD(model.parameters(), lr=lr*10,
                          momentum=0.9, weight_decay=weight_decay)

scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

print("Prepairing data...")




