import time
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备数据，加载数据，准备模型，设置损失函数，
# 设置优化器，开始训练，最后验证，结果聚合展示

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.CIFAR10("../dataset2",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("../dataset2",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集的长度为： {train_data_size}")
print(f"测试数据集的长度为： {test_data_size}")

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
            nn.Softmax(dim =1)
        # 调用nn.Softmax函数，其中dim=1表示对张量的第一个维度进行softmax操作。
        # 返回softmax操作后的张量。
        )
    def forward(self,x):
        x = self.model(x)
        return x

model  = Model()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 100
test_accuracy = 0

writer = SummaryWriter("../../../learn_pytorch/logs")
start_time = time.time()
for i in range(epoch):
    print(f"--------第 {i+1} 轮训练开始---------")

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"此100步训练所用时间为: {end_time-start_time}")
            print(f"训练次数： {total_train_step}，Loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss =0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print(f"整个测试集上的loss： {total_test_loss}")
    print(f"整个测试集上的正确率：{total_test_accuracy}")
    if (total_test_accuracy / test_data_size) > test_accuracy:
        # 此处test_accuracy为上一个epoch的测试集的总体正确率
        # 若此epoch大于上一个epoch的正确率，则保存model
        test_accuracy = total_test_accuracy/test_data_size
        torch.save(model,"model_{}_gpu.pth".format(i+1))

    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_test_accuracy/test_data_size,total_test_step)
    total_test_step += 1

    if i % 99:
        torch.save(model,"model_{}_gpu.pth".format(i+1))
        print("模型已保存")

writer.close()