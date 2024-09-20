'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''

import torch
import sys

if __name__ == '__main__':

    # 从命令行参数获取文件名
    filename = sys.argv[1]

    # 加载文件内容
    data = torch.load(filename)

    # 打印文件内容
    print(data)


