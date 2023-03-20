# -*- coding: utf-8 -*-
# ---
# @File: train.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2023/03/14
# Describe:  using Generto_S2F_V2
# ---
import itertools

from Modules import Generator_F2S, Generator_S2F_v2,Generator_shadow
from Modules import Discriminator
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset_ntire import ImageDataset
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from utils import mask_generator_batch,LambdaLR
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=5, help='size of the batches')
parser.add_argument('--nshadow_root', type=str, default='./ntire23_sr_train_gt', help='root directory of the shadow image')
parser.add_argument('--shadow_root', type=str, default='./ntire23_sr_train_input', help='root directory of the none shaow imag')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to minimum')
parser.add_argument('--offset_epoch', type=int, default=0, help='set of the minimum learing rate')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--C_in', type=int, default=3, help='number of channels of input data')
parser.add_argument('--C_out', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=10, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of saving checkpoint')
parser.add_argument('--save_iter', type=int, default=100, help='save generated image for n iterations')
parser.add_argument('--report_iter', type=int,default=10,help="ever n iters output current loss")
opt = parser.parse_args()
opt.log_path = os.path.join('./output', str(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")) + '.txt')

if not os.path.exists('./output'):
    os.makedirs('./output')

#  model initialization
G_S2N=Generator_S2F_v2(opt.C_in+1, opt.C_out)  # generate none-shadow image
G_S2N_2=Generator_S2F_v2(opt.C_out + 1, opt.C_in) # generate shadow image
G_mask=Generator_shadow(opt.C_in,1)
Dn=Discriminator(opt.C_out) # discriminate none-shadow image


if opt.cuda:
    G_S2N.cuda()
    G_S2N_2.cuda()
    G_mask.cuda()
    Dn.cuda()


# Loss initialization
identity_criterion=nn.L1Loss()
reconstuction_criterion=nn.MSELoss()
discri_criterion=torch.nn.MSELoss() # alternative BCEwithLogitsLoss()

# Optimizer initialization
opt_Dn=torch.optim.Adam(Dn.parameters(), lr=opt.lr, betas=(0.5, 0.999))
opt_GAN=torch.optim.Adam(itertools.chain(G_S2N.parameters(), G_S2N_2.parameters()), lr=opt.lr, betas=(0.5, 0.999))

# Learning Scheduler initialization
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_GAN,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.offset_epoch, opt.decay_epoch).step)
lr_scheduler_Df = torch.optim.lr_scheduler.LambdaLR(opt_Dn,
                                                    lr_lambda=LambdaLR(opt.n_epochs, opt.offset_epoch, opt.decay_epoch).step)


# set all-1 labels for Discriminator-f
target_real = Variable(torch.Tensor(opt.batchSize,1).fill_(1.0), requires_grad=False)
target_fake = Variable(torch.Tensor(opt.batchSize,1).fill_(0.0), requires_grad=False)
mask_non_shadow = Variable(torch.Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(0), requires_grad=False)
mask_all_shadow = Variable(torch.Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(1), requires_grad=False)
if opt.cuda:
    target_real=target_real.cuda()
    target_fake=target_fake.cuda()
    mask_non_shadow=mask_non_shadow.cuda()
    mask_all_shadow=mask_all_shadow.cuda()


# data augmentation
transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
    transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

train_dataloader = DataLoader(ImageDataset(opt.shadow_root,opt.nshadow_root, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


if __name__=="__main__":
    lambda_1=5
    lambda_2=10
    lambda_3=10
    to_pil = transforms.ToPILImage()
    for epoch in range(opt.n_epochs):
        for i,batch in enumerate(train_dataloader):
            shadow_img=Variable(batch['shadow_img'])
            none_shadow=Variable(batch['none_shadow_img'])
            if opt.cuda:
                shadow_img=shadow_img.cuda()
                none_shadow=none_shadow.cuda()

            # Generating pseudo shadow_mask
            pseudo_mask=G_mask(shadow_img)

            # Generator none-shadow image and corresponding Discriminator F
            fake_img=G_S2N(shadow_img,pseudo_mask,pseudo_mask-mask_non_shadow)
            fake_img_discri=Dn(fake_img)
            discri_img_loss=discri_criterion(fake_img_discri,target_real)


            # Generator Shadow and corresponding Discriminator S
            fake_img_2=G_S2N_2(fake_img, pseudo_mask,pseudo_mask-mask_non_shadow)
            fake_img_discri_2=Dn(fake_img_2)
            discri_img_loss_2=discri_criterion(fake_img_discri_2,target_real)

            # real mask
            real_mask=mask_generator_batch(shadow_img, none_shadow)
            
            # identity loss for shadow_img
            same_none=G_S2N_2(none_shadow, real_mask,real_mask-mask_non_shadow)
            identity_loss_shadow=identity_criterion(none_shadow, same_none)

            # consistence loss
            consis_target_loss=reconstuction_criterion(fake_img, none_shadow)
            consis_shadow_loss=reconstuction_criterion(fake_img_2, none_shadow)

            # total loss
            loss_G = discri_img_loss_2 + discri_img_loss + lambda_3*identity_loss_shadow+\
                     lambda_2*consis_shadow_loss+lambda_1*consis_target_loss

            # total backward
            opt_GAN.zero_grad()
            loss_G.backward()
            opt_GAN.step()

            ##############################
            ###### Discriminator f #######
            pred_fake=Dn(fake_img.detach())  # avoid upgrading Generator
            discri_loss_fake=discri_criterion(pred_fake,target_fake)

            pred_real=Dn(none_shadow)
            discri_loss_real=discri_criterion(pred_real,target_real)

            loss_Df=(discri_loss_real+discri_loss_fake)*0.5

            opt_Dn.zero_grad()
            loss_Df.backward()
            opt_Dn.step()


            if (i+1) % opt.save_iter == 0:
                img_fake_A = 0.5 * (fake_img_2.detach().data + 1.0)
                img_fake_A = (to_pil(img_fake_A[0].data.squeeze(0).cpu())) # save the first one
                img_fake_A.save('./output/v3/fake_noneshadow.png')
            if  (i+1)%opt.report_iter == 0:
                log = '[epoch %d][iter %d], [loss_G %.5f], [loss_G_identity %.5f], [loss_G_GAN %.5f],' \
                      '[loss_target_cycle %.5f], [loss_shadow_cycle %.5f]' % \
                      (epoch ,i, loss_G, (identity_loss_shadow), (discri_img_loss+discri_img_loss_2),
                       (consis_target_loss), (consis_shadow_loss))
                print(log)
                open(opt.log_path, 'a').write(log + '\n')

        lr_scheduler_Df.step()
        lr_scheduler_G.step()


        if (epoch + 1) % opt.snapshot_epochs == 0:
            torch.save(G_S2N.state_dict(), ('./output/v3/G_S2N_%d.pth' % (epoch + 1)))
            torch.save(G_S2N_2.state_dict(), ('./output/v3/G_S2N2_%d.pth' % (epoch + 1)))
            torch.save(G_mask.state_dict(),('./output/v3/G_mask_%d.pth' % (epoch + 1)))
            torch.save(Dn.state_dict(), ('./output/v3/Dn_%d.pth' % (epoch + 1)))
        print('Epoch:{}'.format(epoch))








