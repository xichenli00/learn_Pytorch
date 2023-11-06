import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data_image_net", split="train",
#                                        download=True, transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)
# 最新版默认是没有预训练，需要使用预训练设置weights='DEFAULT'
# 最新调用预训练模型权重方法：weights=VGG16_Weights.DEFAULT
vgg16_true = torchvision.models.vgg16( weights = "DEFAULT")
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("../dataset2", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 添加模块
vgg16_true.add_module('add_linear', nn.Linear(1000, 10)) # 直接在后面新建一个名为add_linear的模块
# 在某个模块中添加子模块
vgg16_true.classifier.add_module('7',nn.Linear(1000, 10)) # 在模块classifier后面新建一层

print("\n")
print(vgg16_true)
print("\n")
print(vgg16_false)
# 修改某个模块中的子模块
vgg16_false.classifier[6] = nn.Linear(4096,10) # 修改模块classifier中的第6层
print(vgg16_false)


