# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 准备数据，加载数据，准备模型，设置损失函数，
# 设置优化器，开始训练，最后验证，结果聚合展示

# 只有网络模型，数据（输入，标注），损失函数 这三个有.cuda()
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 定义训练的设备------------------------------------------
device = torch.device("cuda")
# device = torch.device("cuda:0")
train_data = torchvision.datasets.CIFAR10(root="../dataset2", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset2", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()
# -----------------------------------
# tudui = tudui.to(device)
tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# ---------------------------------------
# loss_fn = loss_fn.to(device)
loss_fn.to(device)

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        # -------------------------------------------------
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"此轮训练所用时间：{end_time-start_time}")
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # --------------------------------------------------
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            # argmax 函数是一个求最大值索引的函数。当我们对输出张量（outputs）调用 argmax 函数时，它会返回每个样本在输出向量中概率最大的类别的索引。
            # 具体地说，outputs.argmax(1) 中的参数 1 表示在第二个维度上执行 argmax 操作，即对每个样本的预测概率向量执行 argmax 操作，
            # 返回一个表示每个样本最可能的预测类别索引的一维张量。这个一维张量的长度等于批量大小，每个位置的值为该样本最可能的类别的索引。
            # 在分类问题中，模型的输出通常是一个概率分布向量，每个位置表示一个类别的概率。而我们通常只需要知道最可能的类别，
            # 因此可以使用 argmax 函数获取每个样本最可能的类别的索引。这些索引可以与真实标签进行比较，计算模型的准确率。
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    if i == 29:
        torch.save(tudui, "tudui_{}_gpu.pth".format(i+1))
        print("模型已保存")

# 当准确率还是不高时，可以将学习速率调小一点或训练周期还是不够
writer.close()
