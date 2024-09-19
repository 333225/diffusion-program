'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''

from setting import *
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,time_emb_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),# 改变通道数，不变大小
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # time时刻emb添加到channel
        self.time_emb_liner = nn.Linear(time_emb_size, out_channels)
        self.relu = nn.ReLU()

        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # 改变通道数，不变大小
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x,t_emb):
        x = self.seq1(x)
        t_emb = self.relu(self.time_emb_liner(t_emb)).view(x.size(0),x.size(1),1,1)  # t_emb:(batch_size,out_channels,1,1)
        #通道数不变 大小不变
        return self.seq2(x+t_emb)






