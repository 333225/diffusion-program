'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''

from setting import *
from torch import nn


class TimePositionEncoding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size = emb_size//2
        # range:0,1,2,3[0~3]===[0*(-1)*math.log(10000)/ (self.half_emb_size-1)]
        half_emb = torch.exp(torch.arange(self.half_emb_size) * (constant_coefficient / (self.half_emb_size-1)))
        # 将 half_emb 注册为模型的一个常量缓冲区，使其不会被训练更新，但仍会保存在模型中。固化
        self.register_buffer('half_emb', half_emb)

    def forward(self,t):
        # 调整形状为[batch_size,1], [[],[]]
        t = t.view(t.size(0),1)
        # 加维[[x,x,x,x....],[x,x,x,x....],[x,x,x,x....],[x,x,x,x....]....]
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size)
        half_emb_t = half_emb * t
        embs_t = torch.cat([half_emb_t.sin(), half_emb_t.cos()], dim=-1)
        return embs_t

if __name__ == '__main__':
    time_pos_emb = TimePositionEncoding(8).to(Device)
    t = torch.randint(0,interation_times,(2,)).to(Device)
    print(t)
    embs_t = time_pos_emb(t)
    print(embs_t)

