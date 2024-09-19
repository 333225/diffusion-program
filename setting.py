'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''

'''
    需要导入的包
'''
import torch
import matplotlib.pyplot as plt
import math
from torch import nn


'''
    图片大小
'''
img_size = 256

'''
    加噪参数设置
'''
beta_min = 0.0001
beta_max = 0.02

'''
    时间步数
'''
interation_times = 1000


Device = "cuda" if torch.cuda.is_available() else "cpu"

'''
    时间编码固定系数
'''
constant_coefficient = -1 * math.log(10000)

'''
    训练参数
'''


EPOCH = 200
batch_size = 800


