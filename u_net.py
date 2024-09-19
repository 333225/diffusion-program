'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''
from add_noise_forward import add_noise_forward
from conv_block import *
from encode_time import *
from dataset import *

class Unet(nn.Module):
    def __init__(self,img_channels,channels=[64,128,256,512,1024],time_emb_size = 256):
        super().__init__()
        channels = [img_channels] + channels
        print('图像通道数',img_channels)

        # 时间进行转向量化
        self.time_emb = nn.Sequential(
            TimePositionEncoding(time_emb_size),
            nn.Linear(time_emb_size,time_emb_size),
            nn.ReLU()
        )

        # 卷积层通道数增加一倍
        self.enc_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_convs.append(ConvBlock(channels[i],channels[i+1],time_emb_size))

        # 下采样后，图像大小减少一般
        self.max_pool = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.max_pool.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        #  上采样 通道数减半
        self.deconvs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2],kernel_size=2,stride=2))

        # 上采样后图像大小回复一半
        self.dec_convs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.dec_convs.append(ConvBlock(channels[-i-1],channels[-i-2],time_emb_size))

        # 最后将图像恢复成初始状态的通道
        self.output = nn.Conv2d(channels[1],img_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x,t):
        t_emb = self.time_emb(t)


        residual = []

        for i,conv in enumerate(self.enc_convs):
            x = conv(x,t_emb)
            if i!=len(self.enc_convs)-1:
                residual.append(x)
                x = self.max_pool[i](x)

        for i,deconv in enumerate(self.deconvs):
            x = deconv(x)
            residual_x = residual.pop(-1)
            x = self.dec_convs[i](torch.cat((residual_x,x),dim=1),t_emb)

        return self.output(x)

if __name__ == '__main__':
    batch_x = torch.stack((train_dataset[0], train_dataset[1]), dim=0).to(Device)
    batch_x = batch_x * 2 -1
    batch_t = torch.randint(0, interation_times, size=(batch_x.size(0),)).to(Device)  # size=2
    batch_x_t,batch_noise_t = add_noise_forward(batch_x, batch_t)

    print('batch_x_t',batch_x_t.size())
    print('batch_noise_t',batch_noise_t.size())

    unet = Unet(batch_x_t.size(1)).to(Device)
    batch_predict_noise = unet(batch_x_t,batch_t)
    print('batch_predict_noise',batch_predict_noise.size())