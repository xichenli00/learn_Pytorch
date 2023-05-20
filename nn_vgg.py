import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset2',
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)
writer = SummaryWriter("logs5")
step = 0

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # 64,3,32,32
    # output = torch.reshape(imgs, (1, 1, 1, -1)) 和下行功能相同
    output = torch.flatten(imgs)  # 平铺
    print(output.shape) # 196608

    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    print(output.shape)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
