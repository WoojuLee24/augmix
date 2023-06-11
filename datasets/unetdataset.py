import torch
import numpy as np
import augmentations

class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, dropout=0.35, split='train'):
        self.dataset = dataset
        self.preprocess = preprocess
        self.dropout = dropout
        self.split = split

    def __getitem__(self, i):
        x, _ = self.dataset[i]
        y = self.preprocess(x)
        if self.split == 'train':
            dropout_prob = np.random.uniform(low=0.02, high=self.dropout)
        else:
            dropout_prob = self.dropout / 2

        mask = torch.rand_like(y)
        mask = (mask > dropout_prob).float()
        x = y * mask    # TODO: zero value of mean value for deleted pixels?

        return x, y

    def __len__(self):
        return len(self.dataset)


class UNetAugopDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, aug_severity=3):
        self.dataset = dataset
        self.preprocess = preprocess
        self.aug_severity = aug_severity

    def __getitem__(self, i):
        x, _ = self.dataset[i]

        y = []
        x_copy = x.copy()
        x = self.preprocess(x)
        y.append(x)

        aug_list = augmentations.augmentations
        for op in aug_list:
            x_aug = op(x_copy, self.aug_severity)
            x_aug = self.preprocess(x_aug)
            y.append(x_aug)
        ys = torch.cat(y, dim=0)
        return x, ys

    def __len__(self):
        return len(self.dataset)