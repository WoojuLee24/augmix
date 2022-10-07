# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import numpy as np
import datasets.utils as utils

def pixmix(orig, mixing_pic, preprocess, k, beta, all_ops, aug_severity):
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig, all_ops, aug_severity))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(k + 1)):

        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig, all_ops, aug_severity))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)


def augment_input(image, all_ops, aug_severity):
    aug_list = utils.augmentations_all if all_ops else utils.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


class RandomImages300K(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.dataset = np.load(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img), 0

    def __len__(self):
        return len(self.dataset)


class PixMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform PixMix."""

    def __init__(self, dataset, mixing_set, preprocess, **kwargs):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess

        self.no_jsd = kwargs['no_jsd']
        self.k = kwargs['k'] if 'k' in kwargs else 4 # (int), Mixing iterations
        self.beta = kwargs['beta'] if 'beta' in kwargs else 3 # (int) Severity of mixing
        self.all_ops = kwargs['all_ops'] if 'all_ops' in kwargs else False # Turn on all augmentation operations (+brightness, contrast, color, sharpness).
        self.aug_severity = kwargs['aug_severity'] if 'aug_severity' in kwargs else 3 # (int) Severity of base augmentation operators

    def __getitem__(self, i):
        x, y = self.dataset[i]

        if self.no_jsd:
            rnd_idx = np.random.choice(len(self.mixing_set))
            mixing_pic, _ = self.mixing_set[rnd_idx]
            return pixmix(x, mixing_pic, self.preprocess, self.k, self.beta, self.all_ops, self.aug_severity), y
        else:
            tensorize, normalize = self.preprocess['tensorize'], self.preprocess['normalize']
            original = tensorize(x)

            rnd_idx = np.random.choice(len(self.mixing_set))
            mixing_pic, _ = self.mixing_set[rnd_idx]
            aug1 = pixmix(x, mixing_pic, self.preprocess, self.k, self.beta, self.all_ops, self.aug_severity)

            rnd_idx = np.random.choice(len(self.mixing_set))
            mixing_pic, _ = self.mixing_set[rnd_idx]
            aug2 = pixmix(x, mixing_pic, self.preprocess, self.k, self.beta, self.all_ops, self.aug_severity)

            im_tuple = (original, aug1, aug2)

            return im_tuple, y

    def __len__(self):
        return len(self.dataset)