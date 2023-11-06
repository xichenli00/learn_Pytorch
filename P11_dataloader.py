import torchvision
# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#transform = torchvision.transforms.Compose([torchvision.transforms.Resize([512,512]),torchvision.transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10("./dataset2", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, num_workers=0, shuffle=True, drop_last=False)
# num_works>0在win系统中可能会报错BrokenPipeError
# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape) # chw
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()
