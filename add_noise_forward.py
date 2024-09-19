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
    # 生成随机噪声(形状和传入图像一样)
    batch_noise_t = torch.randn_like(batch_x)

    # 调整当前时间的alphas_bar形状以匹配batch_x
    batch_alpha_bar = alphas_bar.to(Device)[batch_t].reshape(batch_x.size(0),1,1,1)

    # 根据公式计算
    batch_x_t = torch.sqrt(batch_alpha_bar) * batch_x + torch.sqrt(1 - batch_alpha_bar) * batch_noise_t

    # 返回加噪图片和噪声
    return batch_x_t, batch_noise_t


# 测试代码
if __name__ == '__main__':

    '''
        函数原型-
            torch.stack(tensors, dim=0, out=None)
        参数-
            tensors: 需要堆叠的张量序列。所有的张量必须具有相同的形状。
            dim: 新维度的索引。在这个新维度上堆叠张量。默认值是 0。
            out: 可选，输出张量。如果指定了这个参数，结果将存储在 out 中。
        返回值-
            返回一个新的张量，其中的元素是沿指定的维度堆叠的输入张量。
    '''
    batch_x = torch.stack((train_dataset[0],train_dataset[1]), dim=0).to(Device)


    # 绘制原始图像
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil(batch_x[0]))

    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    plt.show()



    # 归一化处理将图片像素映射到[-1,1]
    batch_x = batch_x * 2-1

    '''
        函数原型-
            torch.randint(low, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        参数-
            low: 生成的随机整数的下界（包含该值）。
            high: 生成的随机整数的上界（不包含该值）。
            size: 生成张量的形状。例如，(2, 3) 代表生成一个 2x3 的张量。
            out: 可选，存储结果的张量。
            dtype: 可选，返回张量的数据类型。默认为 torch.int64。
            device: 可选，生成张量的设备（如 cpu 或 cuda）。
            requires_grad: 可选，是否需要计算梯度。默认为 False。
        返回值-
            返回一个填充随机整数的张量，其形状由 size 参数指定，范围是 [low, high)。
    '''
    batch_t = torch.randint(0,interation_times,size = (batch_x.size(0),)).to(Device)# size=2
    print('batch_x',batch_x.size(0))
    print('batch_t',batch_t)

    # 添加噪声
    batch_x_t, batch_noise_t = add_noise_forward(batch_x , batch_t)
    print('batch_x_t',batch_x_t.size())
    print('batch_noise_t',batch_noise_t.size())

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil((batch_x_t[0] + 1)/2))

    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil((batch_x_t[1] + 1)/2))
    plt.show()

