import torch 
from Modules import Generator_S2F_v2,Generator_shadow
from dataset_ntire import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--shadow_root', type=str, default='./ntire23_shrem_test_input_2', help='root directory of the shadow image')
parser.add_argument('--nshadow_root', type=str, default='', help='root directory of the none shaow imag')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=10, help='number of cpu threads to use during batch generation')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--target_h',type=int,default=1440,help="size of output image")
parser.add_argument('--target_w',type=int,default=1920,help="size of output image")
opt = parser.parse_args()

model_stage_1=Generator_S2F_v2()
model_stage_1.load_state_dict(torch.load('.、output/v3/G_S2N_200.pth'))
model_stage_1=model_stage_1.eval()

model_stage_2=Generator_S2F_v2()
model_stage_2.load_state_dict(torch.load('./output/v3/G_S2N2_200.pth'))
model_stage_2=model_stage_2.eval()
 
G_mask=Generator_shadow()
G_mask.load_state_dict(torch.load('./output/v3/G_mask_200.pth'))
G_mask=G_mask.eval()


mask_non_shadow = Variable(torch.Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(0), requires_grad=False)

if opt.cuda:
    model_stage_1.cuda()
    model_stage_2.cuda()
    G_mask.cuda()
    mask_non_shadow=mask_non_shadow.cuda()


transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
    transforms.Resize((opt.size,opt.size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
inverse_tranforms=transforms.Compose([
     transforms.Resize((opt.target_h,opt.target_w))
])

train_dataloader = DataLoader(ImageDataset(opt.shadow_root,opt.nshadow_root, transforms_=transforms_,mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


if __name__=="__main__":
    to_pil = transforms.ToPILImage()
    for i,batch in enumerate(train_dataloader):
            shadow_img=Variable(batch['shadow_img'])
            if opt.cuda:
                shadow_img=shadow_img.cuda()
            fake_mask=G_mask(shadow_img)
            pre_fake_img=model_stage_1(shadow_img,fake_mask,fake_mask-mask_non_shadow)
            fake_img=model_stage_2(pre_fake_img,fake_mask,fake_mask-mask_non_shadow)

            img_fake_none = 0.5 * (fake_img.detach().data + 1.0)
            img_fake_none=inverse_tranforms(img_fake_none)
            img_fake_none = (to_pil(img_fake_none.data.squeeze(0).cpu()))
            img_fake_none.save(f'./output/result/fake_{i}.png')
            # if i < 50:
            #     print(f'finish {i}')
            # else:
            #      break
