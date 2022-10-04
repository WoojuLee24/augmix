import io
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import augmentations

def train_transforms():
    transforms_list = []

    transforms_list.extend([
        transforms.RandomApply([APRecombination()], p=1.0),
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    # else:
    #     transforms_list.extend([
    #         transforms.RandomCrop(32, padding=4, fill=128),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ])

    return transforms_list

class APRecombination(object):   #apr-s
    def __init__(self, img_size=32, aug=None):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''

        op = np.random.choice(self.aug_list) #APR-S
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > 0.5:
            return x        # x: one augmentation apply

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)    # x_aug: two augmentation apply

        x = np.array(x).astype(np.uint8) 
        x_aug = np.array(x_aug).astype(np.uint8)
        
        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        fft_2 = np.fft.fftshift(np.fft.fftn(x_aug))
        
        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
        abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

        fft_1 = abs_1*np.exp((1j) * angle_2)
        fft_2 = abs_2*np.exp((1j) * angle_1)

        p = random.uniform(0, 1)

        if p > 0.5:
            x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        else:
            x = np.fft.ifftn(np.fft.ifftshift(fft_2))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        
        return x


#choice one of below when using jsd loss. it has a same result when not using jsd loss.


# class AprS(torch.utils.data.Dataset):
#     def __init__(self, dataset, apr_p, no_jsd=False):
#         self.dataset = dataset
#         self.no_jsd = no_jsd
#         transforms_list = ([
#             transforms.RandomApply([APRecombination()], p=1.0),
#             transforms.RandomCrop(32, padding=4, fill=128),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])
#         transforms_list_original = ([
#             transforms.RandomCrop(32, padding=4, fill=128),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5] * 3, [0.5] * 3),
#         ])
#         if apr_p == 0:
#             transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
#
#
#         self.train_transform = transforms.Compose(transforms_list)
#         self.train_transform_original = transforms.Compose(transforms_list_original)
#
#     def __getitem__(self, i):
#         x, y = self.dataset[i]
#         if self.no_jsd:
#             return self.train_transform(x.copy()), y
#
#         else:
#             original = self.train_transform_original(x.copy())
#             aug1 = self.train_transform(x.copy())
#             aug2 = self.train_transform(x.copy())
#             im_tuple = (original, aug1, aug2)
#             return im_tuple, y
#
#     def __len__(self):
#         return len(self.dataset)





class AprS(torch.utils.data.Dataset):
    def __init__(self, dataset, apr_p, no_jsd=False):
        self.dataset = dataset
        self.no_jsd = no_jsd
        transforms_list = ([
            transforms.RandomApply([APRecombination()], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        #when using jsd
        transforms_common = ([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
        ])
        transforms_aug = ([transforms.RandomApply([APRecombination()], p=1.0),
                           transforms.ToTensor(),])
        transforms_original = ([transforms.ToTensor(),
                                transforms.Normalize([0.5] * 3, [0.5] * 3)
                                ])

        if apr_p == 0:
            transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))

            transforms_aug.append(transforms.Normalize([0.5] * 3, [0.5] * 3))


        self.train_transform = transforms.Compose(transforms_list)
        self.transforms_common = transforms.Compose(transforms_common)
        self.transforms_aug = transforms.Compose(transforms_aug)
        self.transforms_original = transforms.Compose(transforms_original)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.train_transform(x.copy()), y

        else:
            common = self.transforms_common(x.copy())
            original = self.transforms_original(common.copy())
            aug1 = self.transforms_aug(common.copy())
            aug2 = self.transforms_aug(common.copy())
            im_tuple = (original, aug1, aug2)
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


#apr-p
def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    return mixed_x











