import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3)) # batchsize,h,w,channels

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)  # 差之和2

loss_mse = nn.MSELoss()  # 平方差损失
result_mse = loss_mse(inputs, targets)  # (0+0+2^2)/3  4/3

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
print(x)
print(x.shape)
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)

