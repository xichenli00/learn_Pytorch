# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
# from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn
# 需要初始模型才能加载
model = torch.load(r"E:\PythonProjects\learn_pytorch\vgg16_method1.pth")
print(model)
print("\n")
model = torch.load(r"E:\PythonProjects\learn_pytorch\vgg16_method2.pth")
print(model)
# 方式2，加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load(r"E:\PythonProjects\learn_pytorch\vgg16_method2.pth"))
vgg16.load_state_dict(torch.load(r"E:\PythonProjects\learn_pytorch\vgg16_method2.pth"))

model = torch.load(r"E:\PythonProjects\learn_pytorch\vgg16_method2.pth")
print(model)

# 陷阱1
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load(r'E:\PythonProjects\learn_pytorch\tudui_method1.pth')
# print(model)