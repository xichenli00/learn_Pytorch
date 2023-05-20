import torch


# 所以增加非线性的激活函数实际上是给模型增加非线性的表达能力或者因素，
# 有了非线性函数模型的表达能力就会更强。
# 整个模型就像活了一样，而不是想机器只会做单一的线性操作。
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Sigmoid, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset2", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 现在input可以只有(C，H，W)了，不需要N四维(n,c,h,w)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


# 最大池化： 重点采样
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)
writer = SummaryWriter("logs4")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
# tensorboard --logdir=logs4 --samples_per_plugin=images=1000
