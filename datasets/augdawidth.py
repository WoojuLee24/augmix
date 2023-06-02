import torch
import numpy as np
import augmentations

def aug(image, image2, image3, preprocess, all_ops, da_prob, mixture_width, mixture_depth, aug_severity, mixture_coefficient=1):
    """Perform AugDASet augmentations and ctrain_data, preprocess, args.no_jsdompute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(mixture_coefficient, mixture_coefficient))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        if i == 0:
            image_aug = image2.copy()
        elif i == 1:
            image_aug = image3.copy()
        else:
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)

            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix

    # # # debug
    # mixed_image = (mixed - mixed.min()) / (mixed.max() - mixed.min())
    # mixed_image = mixed_image.cpu().detach().numpy()
    # mixed_image = (mixed_image * 255).astype(np.uint8)
    # mixed_image = np.transpose(mixed_image, (1, 2, 0))
    # import matplotlib.pyplot as plt
    # plt.imsave('/ws/data/log/cifar10/debug/1.png', np.asarray(image))
    # plt.imsave('/ws/data/log/cifar10/debug/2.png', np.asarray(image2))
    # plt.imsave('/ws/data/log/cifar10/debug/3.png', np.asarray(image3))
    # plt.imsave('/ws/data/log/cifar10/debug/mix.png', mixed_image)

    return mixed


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return self.preprocess(x), y

    def __len__(self):
        return len(self.dataset)


class AugDAWidthDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self,
                 dataset,
                 dataset2,
                 dataset3,
                 preprocess,
                 no_jsd=False,
                 all_ops=False,
                 da_prob=1/6,
                 mixture_width=5,
                 mixture_depth=-1,
                 aug_severity=3,
                 mixture_coefficient=1):
        self.dataset = dataset
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.all_ops = all_ops
        self.da_prob = da_prob
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.mixture_coefficient = mixture_coefficient

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x2, y2 = self.dataset2[i]
        x3, y3 = self.dataset3[i]

        if self.no_jsd:
            return aug(x, x2, x3, self.preprocess, self.all_ops, self.da_prob, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient), y
        else:
            original = self.preprocess(x)
            aug1 = aug(x, x2, x3, self.preprocess, self.all_ops, self.da_prob, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient)
            aug2 = aug(x, x2, x3, self.preprocess, self.all_ops, self.da_prob, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient)
            im_tuple = (original, aug1, aug2)
            # im_tuple = (self.preprocess(x), aug(x, self.preprocess),
            #             aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)