# -*- coding: utf-8 -*-
# ---
# @File: Modules.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2023/03/14
# Describe: 
# ---
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange
import einops

class conv_with_bn(nn.Module):
    def __init__(self,C_in,C_out,kernel_size=(4,4),stride=(2,2),padding=(1,1),before=False,ac='ReLU'):
        super(conv_with_bn, self).__init__()
        self.conv=nn.Conv2d(C_in,C_out,kernel_size, stride,padding)
        self.bn=nn.BatchNorm2d(C_out)
        self.before=before
        if hasattr(nn,ac):
            self.ac=getattr(nn,ac)(inplace=True)
        else:
            raise RuntimeError("Wrong activation function!!!")

    def forward(self,x):
        out=self.conv(x)
        if self.before:
            out=self.bn(out)
            out=self.ac(out)
        else:
            out=self.ac(out)
            out=self.bn(out)
        return out

class transconv_with_bn(nn.Module):
    def __init__(self,C_in,C_out,kernel_size=(4,4),stride=(2,2),padding=(1,1),before=False,ac='ReLu'):
        super(transconv_with_bn, self).__init__()
        self.conv=nn.ConvTranspose2d(C_in,C_out,kernel_size, stride, padding)
        self.bn=nn.BatchNorm2d(C_out)
        self.before=before
        if hasattr(nn,ac):
            self.ac=getattr(nn,ac)()
        else:
            raise RuntimeError("Wrong activation function!!!")

    def forward(self,x):
        out=self.conv(x)
        if self.before:
            out=self.bn(out)
            out=self.ac(out)
        else:
            out=self.ac(out)
            out=self.bn(out)
        return out

class Generator_shadow(nn.Module):
    def __init__(self,C_in=3,C_out=1):
        '''
        transfer none-shadow image to shadow mask
        :param C_in: 
        :param C_out: 
        '''
        super(Generator_shadow, self).__init__()
        self.conv0=conv_with_bn(C_in,32)  # H/2,W/2

        self.conv1=conv_with_bn(32,64,ac='LeakyReLU') # H/4,W/4

        self.conv2=conv_with_bn(64,128,ac='LeakyReLU') # H/8,W/8

        self.conv3=conv_with_bn(128,256,ac='LeakyReLU')  # H/16,W/16

        self.transconv4=transconv_with_bn(256,128,ac='ReLU') # H/8,W/8

        self.transconv5=transconv_with_bn(256,64,ac='ReLU') # H/4,W/4

        self.transconv6=transconv_with_bn(128,32,ac='ReLU') # H/2, W/2

        self.transconv7=transconv_with_bn(64,C_out,ac='Sigmoid') # H, W

    def forward(self,x):
        # encoder
        x0=self.conv0(x)
        x1=self.conv1(x0)
        x2=self.conv2(x1)
        x3=self.conv3(x2)

        # decoder
        x4=self.transconv4(x3)

        cat_1=torch.cat([x4,x2],dim=1)
        x5=self.transconv5(cat_1)

        cat_2=torch.cat([x5,x1],dim=1)
        x6=self.transconv6(cat_2)

        cat_3=torch.cat([x6,x0],dim=1)
        out=self.transconv7(cat_3)

        return out

class Generator_S2F(nn.Module):
    def __init__(self,C_in=3,C_out=3):
        '''
        transfer hadow image to none-shadow image
        :param C_in: 
        :param C_out: 
        '''
        super(Generator_S2F, self).__init__()

        self.conv0=conv_with_bn(C_in,64)  # H/2,W/2

        self.conv1=conv_with_bn(64,128,ac='LeakyReLU') # H/4,W/4

        self.conv2=conv_with_bn(128,256,ac='LeakyReLU') # H/8,W/8

        self.conv3=conv_with_bn(256,512,ac='LeakyReLU')  # H/16,W/16

        self.conv4=conv_with_bn(512,512,ac='LeakyReLU')  # H/32,W/32


        self.transconv5=transconv_with_bn(512,512,ac='ReLU') # H/16,W/16


        self.transconv6=transconv_with_bn(1024,256,ac='ReLU') # H/8,W/8

        self.transconv7=transconv_with_bn(512,128,ac='ReLU') # H/4,W/4

        self.transconv8=transconv_with_bn(256,64,ac='ReLU') # H/2, W/2

        self.transconv9=transconv_with_bn(128,C_out,ac='ReLU') # H, W

    def forward(self,x):
        # encoder
        x0=self.conv0(x)
        x1=self.conv1(x0)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)

        # decoder
        x5=self.transconv5(x4)

        cat_1=torch.cat([x5,x3],dim=1)
        x6=self.transconv6(cat_1)

        cat_2=torch.cat([x6,x2],dim=1)
        x7=self.transconv7(cat_2)

        cat_3=torch.cat([x7,x1],dim=1)
        x8=self.transconv8(cat_3)

        cat_4=torch.cat([x8,x0],dim=1)
        out=self.transconv9(cat_4)


        return out


class Generator_S2F_v2(nn.Module):
    def __init__(self,C_in=4,C_out=3):
        '''
        transfer none-shadow image to shadow image
        difference between v2 and original, also using shadow info 
        :param C_in:
        :param C_out:
        '''
        super(Generator_S2F_v2, self).__init__()

        # branch_1, downsample
        self.convb1_0=conv_with_bn(C_in,32)  # H/2,W/2

        self.convb1_1=conv_with_bn(32,64,ac='LeakyReLU') # H/4,W/4

        self.convb1_2=conv_with_bn(64,128,ac='LeakyReLU') # H/8,W/8

        self.convb1_3=conv_with_bn(128,256,ac='LeakyReLU')  # H/16,W/16


        # branch_2, downsample
        self.convb2_0=conv_with_bn(C_in,32)  # H/2,W/2

        self.convb2_1=conv_with_bn(32,64,ac='LeakyReLU') # H/4,W/4

        self.convb2_2=conv_with_bn(64,128,ac='LeakyReLU') # H/8,W/8

        self.convb2_3=conv_with_bn(128,256,ac='LeakyReLU')  # H/16,W/16

        # transformer-like attentino
        self.Q_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.K_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.V_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.k_dim=256
        self.activation=nn.Softmax(dim=-1)

        # upsample
        self.transconv4=transconv_with_bn(256,128,ac='ReLU') # H/8,W/8

        self.transconv5=transconv_with_bn(256,64,ac='ReLU') # H/4,W/4

        self.transconv6=transconv_with_bn(128,32,ac='ReLU') # H/2, W/2

        self.transconv7=transconv_with_bn(64,C_out,ac='ReLU') # H, W

    def forward(self,x,mask,inverse_mask):
        shadow_cover=torch.cat([x,mask],dim=1)
        none_shadow_cover=torch.cat([x,inverse_mask],dim=1)  # 1: shadow region; -1 : none-shadow region

        # encoder branch_1
        xb1_0=self.convb1_0(shadow_cover)
        xb1_1=self.convb1_1(xb1_0)
        xb1_2=self.convb1_2(xb1_1)
        xb1_3=self.convb1_3(xb1_2)

        # encoder branch_2
        xb2_0=self.convb2_0(none_shadow_cover)
        xb2_1=self.convb2_1(xb2_0)
        xb2_2=self.convb2_2(xb2_1)
        xb2_3=self.convb2_3(xb2_2)

        Q=self.Q_mapping(xb1_3)
        K=self.K_mapping(xb2_3)
        V=self.V_mapping(xb2_3)
        out=self.activation(torch.matmul(Q,K.transpose(-1,-2)/(self.k_dim**0.5)))
        out = torch.matmul(out,V)
        out=einops.rearrange(out,'B (H W) C -> B C H W',H=int(np.sqrt(out.shape[1])))


        # decoder
        x4=self.transconv4(out)

        cat_1=torch.cat([x4,xb1_2],dim=1)
        x5=self.transconv5(cat_1)

        cat_2=torch.cat([x5,xb1_1],dim=1)
        x6=self.transconv6(cat_2)

        cat_3=torch.cat([x6,xb1_0],dim=1)
        out=self.transconv7(cat_3)

        return out



class Generator_F2S(nn.Module):
    def __init__(self,C_in=4,C_out=3):
        '''
        transfer none-shadow image to shadow image
        :param C_in:
        :param C_out:
        '''
        super(Generator_F2S, self).__init__()

        # branch_1, downsample
        self.convb1_0=conv_with_bn(C_in,32)  # H/2,W/2

        self.convb1_1=conv_with_bn(32,64,ac='LeakyReLU') # H/4,W/4

        self.convb1_2=conv_with_bn(64,128,ac='LeakyReLU') # H/8,W/8

        self.convb1_3=conv_with_bn(128,256,ac='LeakyReLU')  # H/16,W/16


        # branch_2, downsample
        self.convb2_0=conv_with_bn(C_in,32)  # H/2,W/2

        self.convb2_1=conv_with_bn(32,64,ac='LeakyReLU') # H/4,W/4

        self.convb2_2=conv_with_bn(64,128,ac='LeakyReLU') # H/8,W/8

        self.convb2_3=conv_with_bn(128,256,ac='LeakyReLU')  # H/16,W/16

        # transformer-like attentino
        self.Q_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.K_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.V_mapping=nn.Sequential(
            Rearrange('B C H W -> B (H W) C '),
            nn.Linear(256,256),
        )
        self.k_dim=256
        self.activation=nn.Softmax(dim=-1)

        # upsample
        self.transconv4=transconv_with_bn(256,128,ac='ReLU') # H/8,W/8

        self.transconv5=transconv_with_bn(256,64,ac='ReLU') # H/4,W/4

        self.transconv6=transconv_with_bn(128,32,ac='ReLU') # H/2, W/2

        self.transconv7=transconv_with_bn(64,C_out,ac='ReLU') # H, W

    def forward(self,x,mask,inverse_mask):
        shadow_cover=torch.cat([x,mask],dim=1)
        none_shadow_cover=torch.cat([x,inverse_mask],dim=1)  # 1: shadow region; -1 : none-shadow region

        # encoder branch_1
        xb1_0=self.convb1_0(shadow_cover)
        xb1_1=self.convb1_1(xb1_0)
        xb1_2=self.convb1_2(xb1_1)
        xb1_3=self.convb1_3(xb1_2)

        # encoder branch_2
        xb2_0=self.convb2_0(none_shadow_cover)
        xb2_1=self.convb2_1(xb2_0)
        xb2_2=self.convb2_2(xb2_1)
        xb2_3=self.convb2_3(xb2_2)

        Q=self.Q_mapping(xb1_3)
        K=self.K_mapping(xb2_3)
        V=self.V_mapping(xb2_3)
        out=self.activation(torch.matmul(Q,K.transpose(-1,-2)/(self.k_dim**0.5)))
        out = torch.matmul(out,V)
        out=einops.rearrange(out,'B (H W) C -> B C H W',H=int(np.sqrt(out.shape[1])))


        # decoder
        x4=self.transconv4(out)

        cat_1=torch.cat([x4,xb1_2],dim=1)
        x5=self.transconv5(cat_1)

        cat_2=torch.cat([x5,xb1_1],dim=1)
        x6=self.transconv6(cat_2)

        cat_3=torch.cat([x6,xb1_0],dim=1)
        out=self.transconv7(cat_3)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        self.conv0 = conv_with_bn(input_channels, 64)

        self.conv1 = conv_with_bn(64, 128, ac='LeakyReLU')

        self.conv2 = conv_with_bn(128, 256, ac='LeakyReLU')

        self.conv3 = conv_with_bn(256, 512, ac='LeakyReLU')

        self.conv4 = conv_with_bn(512, 1, ac='LeakyReLU')

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv4(x3)

        out=F.avg_pool2d(out,out.size()[2:]).view(out.size()[0],-1)

        return out


if __name__ == '__main__':
    #BCHW
    size = (3, 3, 256, 256)
    input = torch.Tensor(torch.rand(size))
    #
    # mask=torch.ones((3,1,256,256))
    #
    Gf=Generator_shadow()
    fimg=Gf(input)
    #
    # Gs = Generator_F2S()
    # output = Gs(input)
    # Ds= Discriminator()
    # discri=Ds(output)
    # print(output.shape)
