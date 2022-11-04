import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

class Standardizer(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        return img

class CopyChannel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = img.repeat([3,1,1])
        return img
    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, img):
        img = transforms.functional.adjust_contrast(img = img, contrast_factor=self.contrast)
        img = transforms.functional.adjust_brightness(img=img, brightness_factor=self.brightness)
        return img


    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, gsize=(736,480)):

        flip_and_color_jitter = transforms.Compose([

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=(-45, 45),translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
            transforms.ToTensor(),
            CopyChannel(),
            transforms.RandomApply(
                [ColorJitter(brightness=0.4, contrast=0.8)],
                p=0.8
            ),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=gsize, scale=global_crops_scale, ratio=(0.75, 1.3333),interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            Standardizer(),
            
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=gsize, scale=global_crops_scale, ratio=(0.75, 1.3333),interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            Standardizer(),

        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        lsize = (gsize[0]*0.8,gsize[1]*0.8)
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size=lsize, scale=local_crops_scale, ratio=(0.75, 1.3333), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            Standardizer(),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops