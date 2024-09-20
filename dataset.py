'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''


import os
import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from setting import *




class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # 获取文件夹中所有符合条件的图像文件路径（支持.png, .jpg, .jpeg, .bmp 格式）
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg','bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image




# 定义将PIL图像转换为张量的转换操作
pil_to_tensor = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    # 标准化[0,1]
    transforms.ToTensor()
])

# 定义将张量转换为PIL图像的转换操作
tensor_to_pil = transforms.Compose([
    # [0,1]=>[0,255]
    transforms.Lambda(lambda  t: t*255),

    # 将张量的类型转换为torch.uint8（无符号8位整数）
    transforms.Lambda(lambda t: t.type(torch.uint8)),
    transforms.ToPILImage(),
])

img_folder_path = './image'

train_dataset =CustomImageDataset(
    image_folder=img_folder_path,
    transform=pil_to_tensor
)

if __name__ == '__main__':

    # 创建 DataLoader 以批量加载图片
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    img_tensor = train_dataset[0]

    plt.figure(figsize=(5,5))
    pil_img = tensor_to_pil(img_tensor)
    plt.imshow(pil_img)
    plt.show()
