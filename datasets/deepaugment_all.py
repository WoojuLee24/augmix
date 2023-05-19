import torch
import numpy as np
import augmentations
import os
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import CIFAR10
import random

class DAallDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self,
                 datasets,
                 preprocess,
                 with_augmix=False,
                 no_jsd=False,
                 ):
        self.datasets = datasets

        self.preprocess = preprocess
        self.with_augmix = with_augmix
        self.no_jsd = no_jsd


    def __getitem__(self, i):
        if self.no_jsd:
            x, y = random.choice(self.datasets)[i]
            return x, y

        else:
            augs = []
            for ids, d in enumerate(self.datasets):
                if self.with_augmix and ids == 0:
                    aug = d[i][0]
                    augs.extend(aug)
                aug, y = d[i]
                aug = self.preprocess(aug)
                augs.append(aug)

            return augs, y

    def __len__(self):
        return len(self.datasets[0])

