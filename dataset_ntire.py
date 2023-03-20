# -*- coding: utf-8 -*-
# ---
# @File: dataset_ntire.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2023/03/16
# Describe: 
# ---


import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, shadow_root,none_shadow_root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.shadow_path = sorted(glob.glob(shadow_root + '/*.*'))
        self.none_shadow_path = sorted(glob.glob(none_shadow_root + '/*.*'))
        self.mode=mode

    def __getitem__(self, index):
        shadow_sample = self.transform(Image.open(self.shadow_path[index % len(self.shadow_path)]))
        if self.mode=="train":
            none_shadow_sample = self.transform(Image.open(self.none_shadow_path[index % len(self.none_shadow_path)]))
            return {'shadow_img': shadow_sample, 'none_shadow_img': none_shadow_sample}
        else:
            return {'shadow_img': shadow_sample}

    def __len__(self):
        return len(self.shadow_path)
