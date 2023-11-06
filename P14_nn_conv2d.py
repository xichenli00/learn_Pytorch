import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./dataset2", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    #  output = (n+2p-f)/s +1  p=(f-1)/2
    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
print(tudui)

writer = SummaryWriter("logs2")

step = 0
for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)
    print(imgs.shape)
    # torch.Size([64, 3, 32, 32])
    print(output.shape)
    writer.add_images("input",imgs,step)
    # torch.Size([64, 6, 30, 30])

    output = torch.reshape(output,(-1,3,30 ,30))
    print("output,shape:",output.shape)
    # output,shape: torch.Size([128, 3, 30, 30])
    writer.add_images("output",output,step)


    step += 1