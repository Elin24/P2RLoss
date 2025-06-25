# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn.functional as tF
from torch.utils import data
from PIL import Image
import tqdm
import random

if __name__ == '__main__':
    from utils import NormalSample
else:
    from .utils import NormalSample


class SHHA(data.Dataset):
    def __init__(self, root_path, mode, label_prob=1, protc_path=''):
        self.training = (mode == 'train')
        self.label, self.unlabel = [], []
        
        assert protc_path != '', f"protocol path is invalid: {protc_path}"
        with open(protc_path) as f:
            imgids = f.read().strip().split()
            imgids = set(imgids) # [:int(len(imgids) * label_prob)]
        


        imtype = 'jpg'
        for imgf in os.listdir(os.path.join(root_path, mode + '_data', 'images')):
            if not imgf.endswith(imtype):
                continue
            if (not self.training) or (imgf in imgids):
                    self.label.append(imgf.replace('.' + imtype, ''))
            self.unlabel.append(imgf.replace('.' + imtype, ''))
        
        self.imgpath = os.path.join(root_path, mode + '_data', 'images', '{}' + f'.{imtype}')
        self.dotpath = os.path.join(root_path, mode + '_data', 'new-anno', 'GT_{}.npy')
        
        self.norm_func = NormalSample(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            crop_size=(256, 256),
            train = self.training
        )

        print(f"[training = {self.training}]: {len(self.label)} imgs are labeled &  {len(self.unlabel)} imgs are unlabeled.")

    def __len__(self):
        return  len(self.unlabel) if self.training else len(self.label)

    def __getitem__(self, index):
        if self.training:
            lid = random.choice(self.label) # self.label[index % len(self.label)]
            limg, lseq = self.readLabelSampleFromId(lid)
            uid = self.unlabel[index] # random.choice(self.all_ids)
            uimg, umask = self.readUnlabelSampleFromId(uid)
            return limg, lseq, lid, uimg, umask, uid
        else:
            lid = self.label[index] # random.choice(self.label)
            limg, lseq = self.readLabelSampleFromId(lid)
            return limg, lseq, lid

    def readLabelSampleFromId(self, smpid):
        imgpath = self.imgpath.format(smpid)
        img = Image.open(imgpath).convert('RGB')
        img = self.norm_func.im2tensor(img)

        dotseq = torch.from_numpy(np.load(self.dotpath.format(smpid)))
        dotseq = dotseq[:, :2]
        img, dotseq = self.norm_func.process_lable(img, dotseq)
        return img, dotseq

    def readUnlabelSampleFromId(self, smpid):
        imgpath = self.imgpath.format(smpid)
        img = Image.open(imgpath).convert('RGB')
        wa_img = self.norm_func.im2tensor(img)
        sa_img = self.norm_func.strong_aug(img)

        img = self.norm_func.process_unlabel(torch.cat((wa_img, sa_img), dim=0))
        masks = self.random_mask(img)
        return img, masks

    def random_mask(self, uimgs):
        bsize, _, img_h, img_w = uimgs.shape
        cut_img_mask = torch.ones((bsize, 1, img_h, img_w))
        min_cut, max_cut = 1 / 8, 1 / 4
        for i in range(bsize):
            cut_w = int(img_w * (min_cut + random.random() * (max_cut - min_cut)))
            cut_h = int(img_h * (min_cut + random.random() * (max_cut - min_cut)))
            cut_top = random.randint(0, img_h - cut_h)
            cut_left = random.randint(0, img_w - cut_w)
            cut_bottom, cut_right = cut_top + cut_h, cut_left + cut_w
            cut_img_mask[i, :, cut_top:cut_bottom, cut_left:cut_right] = 0
        return cut_img_mask

    @staticmethod
    def collate_fn(samples):
        if len(samples[0]) > 3:
            limgs, lseqs, lids, uimgs, umask, uids = zip(*samples)
            return torch.cat(limgs, dim=0), sum(lseqs, []), lids, torch.cat(uimgs, dim=0),  torch.cat(umask, dim=0),uids
        else:
            limgs, lseqs, lids = zip(*samples)
            return torch.cat(limgs, dim=0), sum(lseqs, []), lids

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(3, 1, 1)
        std = self.std.to(tensor.device).view(3, 1, 1)
        return tensor * std + mean

if __name__ == '__main__':
    datadir = "/qnap/home_archive/wlin38/crowd/data/ori_data/ShanghaiTech/part_A"
    protc_path = '../../dac_label/sha-5.txt'
    data = SHHA(datadir, 'train', protc_path=protc_path)

    denormal = DeNormalize(
         mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    import cv2

    def cv2fig(tensor, name=None):
        # print(name, ":", tensor.shape)
        limg = denormal(tensor).squeeze(0)
        limg = limg.permute(1, 2, 0).cpu().numpy()
        limg = (limg * 255).astype(np.uint8)
        limg = cv2.cvtColor(limg, cv2.COLOR_RGB2BGR)
        if name is not None:
            cv2.imwrite(name, limg)
        return limg
        
        
    # for j, (limg, lseq, lid, uimg, uid) in enumerate(data):

    #     print(limg.shape, lseq.shape, lid, uimg.shape, uid)
    #     # break
    #     cv2_limg = cv2fig(limg, f'{lid}.png')

    #     for i, (x, y) in enumerate(lseq):
    #         cv2_limg = cv2.circle(cv2_limg, (int(x), int(y)), 1, (0, 0, 255), -1)
    #     cv2.imwrite(f'{lid}.png', cv2_limg)
        
    #     cv2_uimg = cv2fig(uimg[:3], f'{uid}_w.png')
    #     cv2_uimg = cv2fig(uimg[3:], f'{uid}_s.png')
    #     break
    
    # test dataloader
    from torch.utils.data import DataLoader

    loader = DataLoader( data, batch_size = 4, num_workers = 1, pin_memory=False, shuffle = True, collate_fn=SHHA.collate_fn)    
    for i, (limg, lseq, lid, uimg, uid) in enumerate(tqdm.tqdm(loader)):
        print(limg.shape, len(lseq), lseq[0].shape, lid, uid, uimg.shape, uid)
        break
        
