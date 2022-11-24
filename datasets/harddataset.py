import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

from PIL import Image


class HardDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 preprocess,
                 parent_dir='/ws/data/log/cifar10/baselines/augmix_b256_jsdv3_lw12/false_images',
                 ):
        super(HardDataset, self).__init__()
        self.dataset = dataset
        self.preprocess = preprocess
        self.parent_dir = parent_dir

        self.mean = 0.5
        self.std = 0.5
        self.normalize = torchvision.transforms.Normalize([self.mean]*3, [self.std]*3)
        self.denormalize = torchvision.transforms.Normalize([-self.mean]*3, [1/self.std]*3)

        file_list = os.listdir(self.parent_dir)
        self.files = dict()
        for filename in file_list:
            fn = filename.split('_')

            index = int(fn[0][1:])
            corruption = '_'.join(fn[1:-3])
            severity = int(fn[-3][1:])
            target = int(fn[-2][1:])
            pred = int(fn[-1].split('.png')[0][1:])

            if not index in self.files:
                self.files[index] = []
            self.files[index].append({
                'filename': filename,
                'corruption': corruption,
                'severity': severity,
                'target': target,
                'pred': pred,
            })
        # self.files = sorted(self.files.items(), key=lambda item: item[0], reverse=False)

    def __getitem__(self, index):
        if index in self.files.keys():
            files = self.files[index]
            for file in files:
                image_path = os.path.join(self.parent_dir, file['filename'])
                image = Image.open(image_path).convert('RGB')
                # file['image'] = normalize(ToTensor(image)[:3, :, :])
                file['image'] = self.preprocess(image)
        else:
            files = dict()

        x, targets = self.dataset[index]
        # if self.denormalized:
        #     if isinstance(x, tuple):
        #         x1, x2, x3 = x
        #         x1 = self.denormalize(x1)
        #         x2 = self.denormalize(x2)
        #         x3 = self.denormalize(x3)
        #         x = (x1, x2, x3)
        #     else:
        #         x = self.denormalize(x)

        return x, targets, files

    def __len__(self):
        return len(self.dataset)