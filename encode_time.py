'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''
import torch

from setting import *
from torch import nn

'''
    主要用于在模型中生成位置编码，在处理序列数据时。这有助于模型捕捉序列中位置的位置信息
'''

class TimePositionEncoding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.emb_size = emb_size


    def forward(self,t):
        # 计算半个嵌入维度的大雄安
        half_emb_size = self.emb_size//2

        # 用于缩放的常数系数
        emb = constant_coefficient / (half_emb_size - 1)

        # 计算每个嵌入的位置信息
        # torch.arange(5)生成([0,1,2,3,4])张量
        emb = torch.exp(emb * (torch.arange(half_emb_size,dtype=torch.float32)))

        # 将t=>[batch_size,1], 将emb=>[1,half_emb_size]
        # 方便与进行广播
        emb = t[:,None] * emb[None,:]

        # 计算正弦和余弦的嵌入。
        # 前半部分为正弦，后半部分为余弦。进行’列‘拼接为一个完整的
        return torch.cat((emb.sin(),emb.cos()),dim=-1)


# 测试代码
if __name__ == '__main__':
    time_pos_emb = TimePositionEncoding(8).to(Device)
    t = torch.randint(0,interation_times,(2,)).to(Device) #size=(2,)  维度为1，包含两个元素
    print(t)
    embs_t = time_pos_emb(t)
    print(embs_t)

