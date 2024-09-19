'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''
import torch

from setting import *
from u_net import *
from dataset import *

import os

# 检查并创建目录
os.makedirs('./Model', exist_ok=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=4,persistent_workers=True, shuffle=True)

try:
    model = torch.load('./Model/model.pt')
except:
    model = Unet(3).to(Device)

print(model)
# 优化值
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 绝对值
loss_fn = nn.L1Loss()

if __name__ == '__main__':
    model.train()
    for epoch in range(EPOCH):
        last_loss = 0.0
        for batch_x in train_loader:
            # 归一化到[-1,1]
            batch_x = batch_x.to(Device) * 2 -1

            # 随机生成时间t
            batch_t = torch.randint(0, interation_times, size=(batch_x.size(0),)).to(Device)  # size=2

            # 添加噪音
            batch_x_t,batch_noise_t = add_noise_forward(batch_x,batch_t)

            # 预测噪音
            batch_predict_t = model(batch_x_t,batch_t)

            # 计算损失
            loss = loss_fn(batch_predict_t,batch_noise_t)

            # 优化参数
            optimizer.zero_grad()

            # 反向传播优化梯度
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
        print('epoch:{} loss={}'.format(epoch, last_loss))
        torch.save(model.state_dict(), './Model/model.pt.tmp')
        os.replace('./Model/model.pt.tmp','./Model/model.pt')
