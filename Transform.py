from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法   tensor数据类型
# transforms.ToTensor

# 为什么需要Tensor数据类型
# tensor数据类型：包装了神经网络所需要的理论基础参数

# img.size为（宽，高）
# arry_img.shape为（高，宽，通道数）
# arry_img.size为 高x宽x通道数 的总个数HWC

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)
writer = SummaryWriter("logs")

# transform如何使用
tensor_trans = transforms.ToTensor()
# ctrl+p 提示参数
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

# normalize
# std = sqrt((X1-mean)^2+(X2-mean)^2+...)
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([5, 0.5, 0.3], [3, 1, 2])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][-1])
print(img_norm)
writer.add_image("Normalize", img_norm, 1)


# resize
print(img.size) #768,512
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize-> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
writer.add_image("resize", img_resize, 0)
print(img_resize.shape) # torch.Size([3, 512, 512])


# compose resize 2 图片短边缩放至x，长宽比保持不变
trans_resize_2 = transforms.Resize(512)
# Totensor方法会将HWC转换成CHW
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("resize2", img_resize_2)


# randomCrop()
trans_random = transforms.RandomCrop([256, 512])
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("img_crop", img_crop, i)

writer.close()

print(tensor_img)
