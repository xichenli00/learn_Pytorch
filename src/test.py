# 完整的模型验证(测试，demo)套路，利用已经训练好的模型，
# 然后给它提供输入
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../imgs/dog.png"
image = Image.open(image_path)

print(image)
# 因为png是四通道的，除了rgb外，还有一个透明度通道。
# 调用convert保留其颜色通道
# 若图片本来就是三个颜色通道，经过此操作，不变，
# 加上这一步后，可以适应png，jpg各种格式的图片

image = image.convert('RGB')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()
     ])

image = transform(image)
print(image.shape)


# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("tudui_30_gpu.pth", map_location=torch.device('cpu'))
print(model)

# 模型输入要求四维，第一个数是batchsize
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()  # 开启测试，将模型转换为测试类型
with torch.no_grad():  # 这一步可以节约内存和性能
    output = model(image)
print(output) # tensor([[  1.0281,  -7.8479,   8.7832,   5.0053,   4.0859,  10.0867,  -4.6588  5.3077, -10.1664,  -7.3292]])

print(output.argmax(1))  # tensor([5])

