'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''
import matplotlib.pyplot as plt

from dataset import *
from setting import *

betas = torch.sin(torch.linspace(beta_min,beta_max,interation_times))

alphas = 1 - betas

# t的噪音系数
alphas_bar = torch.cumprod(alphas,dim = 0)

# t-1的噪音系数
alphas_bar_prev = torch.empty_like(alphas_bar)
alphas_bar_prev[1:] = alphas_bar[0:interation_times-1]
alphas_bar_prev[0] = 1

# 方差张量
variance = (1-alphas) * (1 - alphas_bar) / (1 - alphas_bar_prev)


'''
    加噪函数
    batch_x:输入一批图片(batch,channel,height,width)
    batch_t:输入一个时刻t(batch_size)
'''
def add_noise_forward(batch_x , batch_t):
    batch_noise_t = torch.randn_like(batch_x)
    batch_alpha_bar = alphas_bar.to(Device)[batch_t].view(batch_x.size(0),1,1,1) #view(调整形状和图像一样)
    batch_x_t = torch.sqrt(batch_alpha_bar) * batch_x + torch.sqrt(1 - batch_alpha_bar) * batch_noise_t
    return batch_x_t, batch_noise_t


if __name__ == '__main__':

    batch_x = torch.stack((train_dataset[0],train_dataset[1]), dim=0).to(Device)

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil(batch_x[0]))
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    plt.show()






    batch_x = batch_x * 2-1
    batch_t = torch.randint(0,interation_times,size = (batch_x.size(0),)).to(Device)# size=2
    print('batch_t',batch_t)

    batch_x_t, batch_noise_t = add_noise_forward(batch_x , batch_t)
    print('batch_x_t',batch_x_t.size())
    print('batch_noise_t',batch_noise_t.size())

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil((batch_x_t[0] + 1)/2))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil((batch_x_t[1] + 1)/2))
    plt.show()

