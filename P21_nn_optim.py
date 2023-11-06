import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear, Conv2d, MaxPool2d, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset2',
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, stride=1, padding=2)
        # self.maxpool1 = MaxPool2d(2)  # 池化层默认stride = kernelsize padding=0
        # self.conv2 = Conv2d(32, 32, 5, stride=1, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()  # 1*1kernel 卷积还是flatten展开？
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


tudui = Tudui()
# output = tudui(input)
# print(output)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01, )
# 这就是批量梯度下降，套了优化器
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)

        optim.zero_grad()
        # backward()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)


# 1.关于前面说的loss怎么和优化器关联
# 3.而你对optim进行step操作时，step就会把它自身反向传播后的结果得用进去
# 4.loss函数在其中只是起到了一个提供梯度的作用，而这个梯度就藏在optim中

