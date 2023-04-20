import torch
import numpy as np
import augmentations


def aug(image, preprocess, set_ops, mixture_width, mixture_depth, aug_severity, mixture_coefficient=1):
    """Perform AugMix augmentations and ctrain_data, preprocess, args.no_jsdompute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    assert set_ops>=0, "set operation to use"
    # ctrl
    image_aug = image.copy()
    op = aug_list[set_ops]
    image_aug = preprocess(op(image_aug, aug_severity))

    return image_aug

def augmix(image, preprocess, set_ops, mixture_width, mixture_depth, aug_severity, mixture_coefficient=1):
    """Perform AugMix augmentations and ctrain_data, preprocess, args.no_jsdompute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    assert set_ops >= 0, "set operation to use"
    # ctrl
    image_aug = image.copy()
    op1 = aug_list[0]
    op2 = aug_list[1]
    op3 = aug_list[2]
    image_aug = 0.4 * preprocess(op1(image_aug, aug_severity)) + \
                0.3 * preprocess(op2(image_aug, aug_severity)) + \
                0.3 * preprocess(op3(image_aug, aug_severity))
    mixed = 0.5 * preprocess(image) + 0.5 * image_aug
    return mixed

    # if all_ops:
    #     aug_list = augmentations.augmentations_all

    # ws = np.float32(np.random.dirichlet([1] * mixture_width))
    # m = np.float32(np.random.beta(mixture_coefficient, mixture_coefficient))
    #
    # mix = torch.zeros_like(preprocess(image))
    #
    # for i in range(mixture_width):
    #     image_aug = image.copy()
    #     depth = mixture_depth if mixture_depth > 0 else np.random.randint(
    #         1, 4)
    #     for _ in range(depth):
    #         op = np.random.choice(aug_list)
    #         image_aug = op(image_aug, aug_severity)
    #     # Preprocessing commutes since all coefficients are convex
    #     mix += ws[i] * preprocess(image_aug)
    #
    # mixed = (1 - m) * preprocess(image) + m * mix
    # return mixed




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


class CtrlAugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self,
                 dataset,
                 preprocess,
                 no_jsd=False,
                 set_ops=0,
                 mixture_width=3,
                 mixture_depth=-1,
                 aug_severity=3,
                 mixture_coefficient=1):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.set_ops = set_ops
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.mixture_coefficient = mixture_coefficient

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess, self.set_ops, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient), y
        else:
            original = self.preprocess(x)
            # aug1 = aug(x, self.preprocess, self.set_ops, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient)
            # aug2 = aug(x, self.preprocess, self.set_ops, self.mixture_width, self.mixture_depth, self.aug_severity, self.mixture_coefficient)
            aug1 = augmix(x, self.preprocess, self.set_ops, self.mixture_width, self.mixture_depth, self.aug_severity,
                       self.mixture_coefficient)
            aug2 = augmix(x, self.preprocess, self.set_ops, self.mixture_width, self.mixture_depth, self.aug_severity,
                       self.mixture_coefficient)
            im_tuple = (original, aug1, aug2)
            # im_tuple = (self.preprocess(x), aug(x, self.preprocess),
            #             aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)