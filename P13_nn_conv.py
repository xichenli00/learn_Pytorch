import torch
import torch.nn.functional as F
# torch.nn.functional类比齿轮，torch.nn类比方向盘
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]
                       ])
input = torch.reshape(input, (1, 1, 5, 5))  # minibatch,in_channels,iH,iW
kernel = torch.reshape(kernel, (1, 1, 3, 3))  # o
print("input.shape:",input.shape)
print("kernel.shape:",kernel.shape)

output = F.conv2d(input, kernel, stride=1, padding=0)
print("output:",output)

output2 = F.conv2d(input, kernel, stride=2, padding=0)
print("output2:",output2)

#
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print("output3:",output3)
