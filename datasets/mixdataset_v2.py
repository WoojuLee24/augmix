import torch
import numpy as np
import augmentations


def aug_v2_0(image, preprocess, all_ops, mixture_width, mixture_depth, aug_severity, mixture_alpha, mixture_beta, mixture_fix=False):
    """Perform AugMix augmentations and ctrain_data, preprocess, args.no_jsdompute mixture.
    mixture coefficient -> mixture_alpha, mixture_beta

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))

    if mixture_fix:
        m = mixture_alpha
    else:
        m = np.float32(np.random.beta(mixture_alpha, mixture_beta))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed, m


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


class AugMixDataset_v2_0(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self,
                 dataset,
                 preprocess,
                 no_jsd=False,
                 all_ops=False,
                 mixture_width=3,
                 mixture_depth=-1,
                 aug_severity=3,
                 mixture_alpha=1,
                 mixture_beta=1,
                 mixture_fix=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.all_ops = all_ops
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.mixture_alpha = mixture_alpha
        self.mixture_beta = mixture_beta
        self.mixture_fix = mixture_fix

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug_v2_0(x, self.preprocess, self.all_ops, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_alpha, self.mixture_beta), y
        else:
            original = self.preprocess(x)
            aug1, m1 = aug_v2_0(x, self.preprocess, self.all_ops,
                                self.mixture_width, self.mixture_depth, self.aug_severity,
                                self.mixture_alpha, self.mixture_beta, self.mixture_fix)
            aug2, m2 = aug_v2_0(x, self.preprocess, self.all_ops,
                                self.mixture_width, self.mixture_depth, self.aug_severity,
                                self.mixture_alpha, self.mixture_beta, self.mixture_fix)
            im_tuple = (original, aug1, aug2)
            # mixed_tuple = (0, m1, m2)

            return im_tuple, y  # ,mixed_tuple

    def __len__(self):
        return len(self.dataset)
