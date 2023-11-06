from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2
# opencv是按BGR读的数据记得转换成RGB!!!!!!：cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)
writer.add_image("test", img_array, 1, dataformats='HWC')

# scalar_value (float or string/blobname): Value to save  y轴
# global_step (int): Global step value to record  x轴
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()