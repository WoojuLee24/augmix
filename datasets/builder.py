import os
import torch
from torchvision import transforms
from torchvision import datasets

import augmentations
from .mixdataset import BaseDataset, AugMixDataset
from .mixdataset_v2 import AugMixDataset_v2_0
from .prime.prime import GeneralizedPRIMEModule, PRIMEAugModule, TransformLayer
from .prime.diffeomorphism import Diffeo
from .prime.rand_filter import RandomFilter
from. prime.color_jitter import RandomSmoothColor
from .pixmix import RandomImages300K, PixMixDataset
from .APR import AprS
import utils

def build_dataset(args, corrupted=False):
    dataset = args.dataset
    if (dataset == 'cifar10') or (dataset == 'cifar100'):
        mean, std = [0.5] * 3, [0.5] * 3
    else: # imagenet
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

    if (dataset == 'cifar10') or (dataset == 'cifar100'):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4)])
        test_transform = preprocess

        mixing_set_transform = transforms.Compose([transforms.Resize(36),
                                                   transforms.RandomCrop(32)])

    else: #imagenet
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip()])
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             preprocess])

        mixing_set_transform = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomCrop(224)])
        utils.IMAGE_SIZE = 224

    parent_dir = '/ws/data'
    dir_name = 'cifar' if (dataset == 'cifar10' or dataset == 'cifar100') else dataset
    root_dir = f'{parent_dir}/{dir_name}'

    if corrupted:
        corr1_data = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
        corr2_data = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
        return corr1_data, corr2_data
    else:
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root_dir, train=True, transform=train_transform, download=True)
            train_dataset_apr = datasets.CIFAR10(root_dir, train=True, download=True)
            test_dataset = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
            base_c_path = f'{root_dir}/CIFAR-10-C/'
            num_classes = 10
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root_dir, train=True, transform=train_transform, download=True)
            test_dataset = datasets.CIFAR100(root_dir, train=False, transform=test_transform, download=True)
            base_c_path = f'{root_dir}/CIFAR-100-C/'
            num_classes = 100
        else: # imagenet
            traindir = os.path.join(root_dir, 'train')
            valdir = os.path.join(root_dir, 'val')
            train_dataset = datasets.ImageFolder(traindir, train_transform)
            test_dataset = datasets.ImageFolder(valdir, test_transform)
            base_c_path = f'{root_dir}-c/'
            num_classes = 1000 if dataset == 'imagenet' else 100
            augmentations.IMAGE_SIZE = 224

            # traindir = os.path.join(args.clean_data, 'train')
            # valdir = os.path.join(args.clean_data, 'val')
            # train_dataset = datasets.ImageFolder(traindir, train_transform)
            # # train_dataset = AugMixDataset(train_dataset, preprocess)
            # test_dataset = datasets.ImageFolder(valdir, test_transform)
            # base_c_path = None
            # num_classes = None

        aug = args.aug
        no_jsd = args.no_jsd
        prime_module = None
        if aug == 'none':
            train_dataset = BaseDataset(train_dataset, preprocess, no_jsd)
        elif aug == 'augmix':
            train_dataset = AugMixDataset(train_dataset, preprocess, no_jsd,
                                          args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity, args.mixture_coefficient)
        elif aug == 'augmix_v2.0':
            train_dataset = AugMixDataset_v2_0(train_dataset, preprocess, no_jsd,
                                          args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity,
                                          args.mixture_alpha, args.mixture_beta, args.mixture_fix)
        elif aug == 'prime':
            preprocess = transforms.Compose([transforms.ToTensor()])
            train_dataset = BaseDataset(train_dataset, preprocess)

            # prime aug init
            augmentations = []

            if args.prime.enable_aug.diffeo:
                diffeo = Diffeo(
                    sT=args.prime.diffeo.sT, rT=args.prime.diffeo.rT,
                    scut=args.prime.diffeo.scut, rcut=args.prime.diffeo.rcut,
                    cutmin=args.prime.diffeo.cutmin, cutmax=args.prime.diffeo.cutmax,
                    alpha=args.prime.diffeo.alpha, stochastic=True
                )
                augmentations.append(diffeo)

            if args.prime.enable_aug.color_jit:
                color = RandomSmoothColor(
                    cut=args.prime.color_jit.cut, T=args.prime.color_jit.T,
                    freq_bandwidth=args.prime.color_jit.max_freqs, stochastic=True
                )
                augmentations.append(color)

            if args.prime.enable_aug.rand_filter:
                filt = RandomFilter(
                    kernel_size=args.prime.rand_filter.kernel_size,
                    sigma=args.prime.rand_filter.sigma, stochastic=True
                )
                augmentations.append(filt)

            prime_module = GeneralizedPRIMEModule(
                preprocess=TransformLayer(mean, std),
                mixture_width=args.prime.augmix.mixture_width,
                mixture_depth=args.prime.augmix.mixture_depth,
                no_jsd=args.prime.augmix.no_jsd, max_depth=3,
                aug_module=PRIMEAugModule(augmentations),
            )


        elif aug == 'pixmix':
            if args.use_300k:
                mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip()]))
            else:
                # mixing_set_transform = transforms.Compose([transforms.Resize(36),
                #                                            transforms.RandomCrop(32)])
                mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
            to_tensor = transforms.ToTensor()
            # normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
            normalize = transforms.Normalize(mean, std)
            train_dataset = PixMixDataset(train_dataset, mixing_set, {'normalize': normalize, 'tensorize': to_tensor},
                                          no_jsd=no_jsd, k=args.k, beta=args.beta, all_ops=args.all_ops, aug_severity=args.aug_severity)
        elif aug == 'apr_s':
            train_dataset = AprS(train_dataset_apr, args, no_jsd)

        return train_dataset, test_dataset, num_classes, base_c_path, prime_module


def build_dataloader(train_dataset, test_dataset, args):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    return train_loader, test_loader