# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("../../pytorch-tutorial-master-2/Code")
from Code.utils.utils import MyDataset, validate, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime

train_txt_path = os.path.join("../../pytorch-tutorial-master-2/Code", "..", "Data", "train.txt")
valid_txt_path = os.path.join("../../pytorch-tutorial-master-2/Code", "..", "Data", "valid.txt")

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log
result_dir = os.path.join("../../pytorch-tutorial-master-2/Code", "..", "Result")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # 第二个处理是 transforms.ToTensor()
    # 在这里会对数据进行 transpose，原来是 h*w*c，会经过 img = img.transpose(0,
    # 1).transpose(0, 2).contiguous()，变成 c*h*w 再除以 255，使得像素值归一化至[0-1]之间，
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络------------------------------------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # normal 正态分布；uniform 均匀分布
                # conv2d默认kaiming_uniform 初始化
                # 可借鉴，卷积层使用kaiming_normal ,FC层采用xavier_normal
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


net = Net()     # 创建一个网络
net.to(device)
net.initialize_weights()    # 初始化权值
# ================================ #
#        finetune 权值初始化
#   第一步：保存模型，拥有一个预训练模型；
#   第二步：加载模型，把预训练模型中的权值取出来；
#   第三步：初始化，将权值对应的“放”到新模型中
# ================================ #

# load params 二、加载模型仅仅只是加载模型的参数
pretrained_dict = torch.load('../../Result/11-13_11-45-58/net_params.pkl')

# 三、 初始化
# 获取当前网络的dict 3.1 创建新模型，并且获得新模型的参数字典 net_state_dict
net_state_dict = net.state_dict()

# 剔除不匹配的权值参数 3.2 将pretrained_dict 里不属于net_state_dict 的键剔除掉
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

# 更新新模型参数字典   3.3 用预训练模型的参数字典 对 新模型的参数字典 net_state_dict 进行更新
net_state_dict.update(pretrained_dict_1)

# 将包含预训练模型参数的字典"放"到新模型中 3.4 将更新了参数的字典 “放”回到网络中
net.load_state_dict(net_state_dict)



# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器,dampending阻尼，抑制
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
best_acc = 0.0
for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        # Variable是PyTorch中的一个过时概念，
        # 用于将张量包装成可计算梯度的对象。在较新的PyTorch版本（1.0及更高版本）中，
        # 不再需要使用Variable，
        # 因为张量本身就具备了自动微分的功能，而且计算梯度的方式也更加直接和方便
        inputs, labels = Variable(inputs), Variable(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        # 验证
        net.eval()
        for i, data in enumerate(valid_loader):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = net(images)
            outputs.detach_()
            # 阻止梯度传播：通过调用 detach_()，你可以将一个张量从计算图中分离出来，
            # 这意味着与这个张量相关的操作不会再对梯度进行反向传播。这对于一些情况很有用，
            # 例如在训练过程中冻结某些层的参数，只更新部分模型参数。

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        # 如果模型性能提升，在这个epoch保存一个检查点
        if (conf_mat.trace() / conf_mat.sum()) > best_acc:
            best_acc = conf_mat.trace() / conf_mat.sum()
            print(f"saving model with acc: {best_acc}")
            net_save_path = os.path.join(log_dir,'cur_best_val_acc.pkl')
            torch.save(net.state_dict(),net_save_path)

        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)

print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
# 如果没有验证集，保存最后一个epoch的模型参数
if len(valid_loader) == 0:
    net_save_path = os.path.join(log_dir, 'net_params.pkl')
    torch.save(net.state_dict(), net_save_path)

# 对一批数据进行预测，返回混淆矩阵以及Accuracy
conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)