'''
@-*- coding: utf-8 -*-
@ python：python 3.12
@ 创建人员：LW
@ 创建时间：2024/9/18
'''
import torch

from setting import *
from dataset import *
from add_noise_forward import *
def backward_denoise(model,batch_x_t):
    steps = [batch_x_t,]

    global alphas, alphas_bar, variance

    model = model.to(Device)

    batch_x_t = batch_x_t.to(Device)
    alphas = alphas.to(Device)
    alphas_bar = alphas_bar.to(Device)
    variance = variance.to(Device)

    with torch.no_grad():
        for t in range(interation_times-1,-1,-1):
            batch_t = torch.full((batch_x_t.size(0),),t).to(Device)

            batch_predict_noise = model(batch_x_t,batch_t)

            shape = (batch_x_t.size(0),1,1,1)

            batch_mean_t = 1 / torch.sqrt(alphas[batch_t].view(*shape))* \
                         (
                             batch_x_t -
                             (1-alphas[batch_t].view(*shape))/torch.sqrt(alphas_bar[batch_t].view(*shape)) * batch_predict_noise
                         )
            if t!=0:
                batch_x_t = batch_mean_t + \
                            (
                                torch.randn_like(batch_x_t) * torch.sqrt(variance[batch_t].view(*shape))

                            )
            else:
                batch_x_t = batch_mean_t

            batch_x_t = torch.clamp(batch_x_t,-1.0,1.0).detach()
            steps.append(batch_x_t)

    return steps


if __name__ == '__main__':
    model = torch.load('model.pt')
    batch_size = 2
    batch_x_t = torch.randn(size=(batch_size,3,img_size,img_size))

    steps = backward_denoise(model,batch_x_t)

    num_imgs = 10
    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(0,num_imgs):
            index = int(interation_times/num_imgs)*(i+1)

            final_img = (steps[index][b].to('cpu')+1)/2

            final_img = tensor_to_pil(final_img)

            plt.subplot(batch_size,num_imgs,b*num_imgs+i+1)
            plt.imshow(final_img)

    plt.show()








