# -*- coding: utf-8 -*-

from random import random
from torchvision import transforms
import torch
import torch.nn.functional as F
import random
from PIL import ImageFilter, Image

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class NormalSample(object):
    def __init__(self, mean, std, crop_size=(256, 256), resize_factor=0.3, train=False):
        self.half_h, self.half_w = crop_size
        self.train = train
        self.scale_factor = resize_factor

        self.im2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])

        self.strong_aug  = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.25),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.8),
            self.im2tensor
        ])

    def process_lable(self, image, dotseq):
        if self.train:
            images, dotseqs = self.crop_and_resize(image, dotseq)
        else:
            images, dotseqs = image.unsqueeze(0), [dotseq]

        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            h, w = images.shape[-2:]

        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                    images[i] = torch.flip(images[i], dims=(-1,))
                    dotseqs[i][:, 0] = w - dotseqs[i][:, 0] - 1



        for i, seq in enumerate(dotseqs):
            u = self.nearest(seq)
            dotseqs[i] = torch.cat((seq[:, [1, 0]], u), dim=1)
        # print("L60", dotseqs[0].shape, images.shape)

        return images, dotseqs
    
    def process_unlabel(self, image):
        if self.train:
            images = self.crop_and_resize(image)
        else:
            raise NotImplementedError("Should not happen...")

        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            h, w = images.shape[-2:]

        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                    images[i] = torch.flip(images[i], dims=(-1,))

        return images
    
    def crop_and_resize(self, image, dotseq=None, num_patches=1):
        '''
        image.type = torch.FloatTensor with shape of (3, H, W)
        dotseq.type = torch.FloatTensor with shape of (N, 2)
        '''
        imh, imw = image.shape[-2:]
        
        scale = random.random() * (self.scale_factor * 2) + (1 - self.scale_factor)
        # scale = 2 ** (random.random() * self.scale_factor * 2 - self.scale_factor)
        crop_h = int(self.half_h / scale + 0.5)
        crop_w = int(self.half_w / scale + 0.5)
        
        # print("img:", image.shape, crop_h, crop_w, imh, imw, scale)

        if crop_h > imh or crop_w > imw:
            padw = crop_w - imw
            padh = crop_h - imh
            image = F.pad(image, (0, padw, 0, padh), mode='constant', value=0)
            # dotseq[:, 0] += padw
            # dotseq[:, 1] += padh
            imh, imw = image.shape[-2:]
        # print(image.shape)
        
        crop_imgs, crop_dots = [], []
        rh, rw = self.half_h / crop_h, self.half_w / crop_w
        for _ in range(num_patches):
            start_h = random.randint(0, imh - crop_h)
            start_w = random.randint(0, imw - crop_w)
            end_h = start_h + crop_h
            end_w = start_w + crop_w
            crop_img = image[:, start_h:end_h, start_w:end_w]
            crop_imgs.append(crop_img)

            if dotseq is not None:
                idx = (dotseq[:, 0] >= start_w) & (dotseq[:, 0] <= end_w) & (dotseq[:, 1] >= start_h) & (dotseq[:, 1] <= end_h)
                selected_dot = dotseq[idx]
                selected_dot[:, 0] = (selected_dot[:, 0] - start_w) * rw
                selected_dot[:, 1] = (selected_dot[:, 1] - start_h) * rh
            

                crop_dots.append(selected_dot)

        crop_imgs = torch.stack(crop_imgs, dim=0)
        crop_imgs = F.interpolate(crop_imgs, (self.half_h, self.half_w), mode='bilinear', align_corners=False)
        
        if dotseq is not None:
            return crop_imgs, crop_dots
        else:
            return crop_imgs

    def nearest(self, seq):
        seqlen = seq.size(0)
        if seqlen <= 1:
            return torch.zeros(seqlen, 1) + 32
        xx = (seq ** 2).sum(dim=-1, keepdim=True)
        xy = seq @ seq.T
        yy = xx.T
        L2 = (xx - 2 * xy + yy).relu()
        m = torch.kthvalue(L2, 2, dim=1).values
        return m.view(-1, 1)
    
def jpg2id(jpg):
    return jpg.replace('.jpg', '')