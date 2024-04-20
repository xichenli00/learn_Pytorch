# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 准备数据，加载数据，准备模型，设置损失函数，
# 设置优化器，开始训练，最后验证，结果聚合展示
# val_acc = total_val_correct / len(val_dataset)
# train_loss = loss_gima / len(train_loader)
import sys

import numpy as np
# 只有网络模型，数据（输入，标注），损失函数 这三个有.cuda()
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.utils.utils import MyDataset, validate, show_confMat
import time
# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import os
import datetime

# 定义超参数
batch_size = 16
learning_rate = 1e-3
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_val_step = 0  # 记录测试的次数
epochs = 100  # 训练的轮数
val_best_accuracy = 0  # 测试集整体正确率

classes_name = ['normal', 'tumor']  # 定义类别名称
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # 定义训练的设备

# -----------------------log-添加tensorBoard-------------------------------------------------------
result_dir = os.path.join("..", "..", "Result")
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)
# -------------------------------------------------------------------------------

# ------------------------------- 加载数据 ----------------------------------------------
# 数据预处理设置;先计算数据集中的均值和方差
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]

normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# # 打标签
# labels = ['normal', 'tumor']
# labels_to_id = dict((c, i) for i, c in enumerate(labels))
# print(labels_to_id)
# id_to_label = dict((v, k) for k, v in labels_to_id.items())
# all_labels = []
# # 对所有图片路径进行迭代
# for img in all_imgs_path:
#     # 区分出每个img，应该属于什么类别
#     for i, c in enumerate(labels):
#         if c in img:
#             all_labels.append(i)
# # print(all_labels)  # 得到所有标签
# print(len(all_labels))

# 划分测试集和训练集
# index = np.random.permutation(len(all_imgs_path))
# 
# all_imgs_path = np.array(all_imgs_path)[index]
# all_labels = np.array(all_labels)[index]
# 
# # 90% as train
# s = int(len(all_imgs_path) * 0.9)
# print("训练数据：".format(s))
# 
# train_imgs = all_imgs_path[:s]
# train_labels = all_labels[:s]
# print(train_labels)
# val_imgs = all_imgs_path[s:]
# val_labels = all_labels[s:]
# print(val_labels)


# 构建MyDataset实例
# train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
# valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

train_data = torchvision.datasets.CIFAR10(root="../dataset2", train=True,
                                          transform=trainTransform,
                                          download=True)
val_data = torchvision.datasets.CIFAR10(root="../dataset2", train=False,
                                        transform=validTransform,
                                        download=True)

# length 长度
train_data_size = len(train_data)
val_data_size = len(val_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(val_data_size))

# 设置工作线程
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
          8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=nw,
                              shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=nw)


# --------------------------------定义网络---------------------------------------------
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
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)  # # 添加softmax层，dim=1表示在类别维度进行softmax操作
        )

    def forward(self, x):
        x = self.model(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# ------------------模型初始化，权重初始化，权重微调(参考train_霹雳吧啦和main)------------------------------------------------------------
tudui = Tudui()
tudui.initialize_weights()
# -----------------------------------
# load pretrain weights
# download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
"""
# ================================ #
#        finetune 权值初始化
#   第一步：保存模型，拥有一个预训练模型；
#   第二步：加载模型，把预训练模型中的权值取出来；
#   第三步：初始化，将权值对应的“放”到新模型中
# ================================ #
"""

# # load params 二、加载模型仅仅只是加载模型的参数
# pretrained_dict = torch.load('net_params.pkl')
#
# # 三、 初始化
# # 获取当前网络的dict 3.1 创建新模型，并且获得新模型的参数字典 net_state_dict
# net_state_dict = net.state_dict()
#
# # 剔除不匹配的权值参数 3.2 将pretrained_dict 里不属于net_state_dict 的键剔除掉
# pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
#
# # 更新新模型参数字典   3.3 用预训练模型的参数字典 对 新模型的参数字典 net_state_dict 进行更新
# net_state_dict.update(pretrained_dict_1)
#
# # 将包含预训练模型参数的字典"放"到新模型中 3.4 将更新了参数的字典 “放”回到网络中
# net.load_state_dict(net_state_dict)

tudui.to(device)
#  ---------------------定义损失函数和优化器-------------------------------------------------------------------
# 此处可设置不同类别的权重！！！！weights = torch.tensor([34.0, 1.0]).cuda()
loss_fn = nn.CrossEntropyLoss(weight=None)
loss_fn.to(device)

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
# params = [p for p in tudui.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate, momentum=0.9,
                            dampening=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                            gamma=0.1)  # 设置学习率下降策略
# optimizer = torch.optim.Adam(tudui.parameters(),lr=0.0001)


# ---------------------------------训练过程---------------------------------------------------------
start_time = time.time()
# train_steps = len(train_dataloader)
for epoch in range(epochs):

    loss_sigma = 0.0  # 一个epoch的loss之和
    correct = 0.0
    total = 0.0  # 已训练的图像个数
    scheduler.step()  # 更新学习率

    print("-------第 {} 轮训练开始-------".format(epoch + 1) + "\n")

    # 训练步骤开始
    tudui.train()
    train_bar = tqdm(train_dataloader,file=sys.stdout)
    for i, data in enumerate(train_bar):
        # start_time2 = time.time()
        imgs, targets = data
        # -------------------------------------------------
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()  # optimizer.zero_grad() 零梯度具体放置在哪
        outputs = tudui(imgs)  # 前向传播
        loss = loss_fn(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化器，参数更新

        total_train_step = total_train_step + 1  # 多少个训练步骤，每个步骤都包含了对一个小批次数据的处理;len(train_dataloader)
        total += len(targets) # targets.size(0) # 多少个图像个数;len(train_dataset)
        # end_time2 = time.time()
        # print("\r训练进度：{}/{}   一个batchsize的时间：{}".format(total, train_data_size,end_time2-start_time2), end="")

        # 统计预测信息
        correct += (outputs.argmax(1) == targets).sum().item()
        # _, predicted = torch.max(outputs.data, 1)
        # total +=targets.size(0)   # len(targets)
        # correct += (predicted == targets).squeeze().sum().numpy()
        loss_sigma += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs, loss)
        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, epochs, i + 1, len(train_dataloader), loss_avg, correct / total))

            """
            在一个图表中记录多个标量的变化，常用于对比，如 trainLoss 和 validLoss 的比较等
            main_tag(string)- 该图的标签。
            tag_scalar_dict(dict)- key 是变量的 tag，value 是变量的值。
            global_step(int)- 曲线图的 x 坐标
            walltime(float)- 为 event 文件的文件名设置时间，默认为 time.time()
            """
            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], i)

        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"此100个训练步骤所用时间：{end_time - start_time}")
            print("训练次数：{}, Loss: {}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

            # 每个epoch，记录梯度，权值
    # 每个epoch，记录梯度，权值
    for name, layer in tudui.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(),epoch)


    # ----------------------------验证/测试过程------------------------------------------------------------------------
    tudui.eval()
    total_val_loss = 0
    total_accuracy = 0
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
    with torch.no_grad():
        val_bar = tqdm(val_dataloader, file=sys.stdout)

        for data in val_bar:
            imgs, targets = data
            # --------------------------------------------------
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)

            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()

            correct = (outputs.argmax(1) == targets).sum().item()
            total_accuracy += correct

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)


            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # 统计混淆矩阵
            for j in range(len(targets)):
                cate_i = targets[j].numpy()
                pre_i = predicted[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0
                
        print('{} set Accuracy:{:.2%}'.format('Valid',conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group',
                           {'valid_loss': total_val_loss / len(val_dataloader)},epoch)
        writer.add_scalars('Accuracy_group',
                           {'valid_acc': conf_mat.trace() / conf_mat.sum()},epoch)

    val_acc = total_accuracy / val_data_size
    print("本周期整体测试集上的Loss: {}".format(
        total_val_loss / len(val_dataloader)))
    print("本周期整体测试集上的正确率: {}".format(val_acc))

    writer.add_scalar("val_loss", total_val_loss, total_val_step)
    # 感觉此处准确率不准，total_val_step不适合作横坐标
    writer.add_scalar("val_accuracy", val_acc, total_val_step)
    total_val_step = total_val_step + 1

    if val_best_accuracy < val_acc:
        val_best_accuracy = val_acc
        torch.save(tudui, "tudui_{}_gpu.pth".format(epoch + 1))
    # if epoch == 99:
    #     torch.save(tudui, "tudui_{}_gpu.pth".format(epoch + 1))
    #     print("第100个epoch的模型已保存")
print("Finished Training \n")
# 当准确率还是不高时，可以将学习速率调小一点或训练周期还是不够
writer.close()

# --------------绘制混淆矩阵图--------------------------------------------------------------
conf_mat_train, train_acc = validate(tudui, train_dataloader, 'train',
                                     classes_name)
conf_mat_valid, valid_acc = validate(tudui, val_dataloader, 'valid',
                                     classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
