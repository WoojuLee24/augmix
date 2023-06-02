import torch
import numpy as np
import augmentations
import os
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import CIFAR10


class DADataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self,
                 dataset1,
                 dataset2,
                 dataset3,
                 preprocess,
                 with_augmix=False,
                 no_jsd=False,
                 ):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

        self.preprocess = preprocess
        self.with_augmix = with_augmix
        self.no_jsd = no_jsd


    def __getitem__(self, i):
        if self.no_jsd:
            if np.random.uniform() < 0.5:
                x, y = self.dataset2[i]
            else:
                x, y = self.dataset1[i]

            return x, y

        else:
            cae = self.dataset2[i][0]
            edsr = self.dataset3[i][0]

            cae = self.preprocess(cae)
            edsr = self.preprocess(edsr)

            original, y = self.dataset1[i]
            if not self.with_augmix:
                original = self.preprocess(original)
                im_tuple = (original, cae, edsr)    # original is CIFAR10 original
            else:
                im_tuple = original + (cae, edsr)   # original is composed of 3-tuple augmix


            return im_tuple, y

    def __len__(self):
        return len(self.dataset1)
