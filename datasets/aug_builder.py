import os
import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np

import augmentations
# from .mixdataset import BaseDataset, AugMixDataset
from .unetdataset import UNetDataset, UNetAugopDataset
from .prime.prime import GeneralizedPRIMEModule, PRIMEAugModule, TransformLayer
from .prime.diffeomorphism import Diffeo
from .prime.rand_filter import RandomFilter
from. prime.color_jitter import RandomSmoothColor
from .pixmix import RandomImages300K, PixMixDataset
from .deepaugment import DADataset
from .deepaugment_all import DAallDataset
from .augdaset import AugDASetDataset
from .augdawidth import AugDAWidthDataset
from .APR import AprS
import utils


def augbuild_dataset(args, corrupted=False):
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
            # test_dataset = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
            test_dataset = datasets.CIFAR10(root_dir, train=False, download=True)
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

        aug_task = args.aug_task
        prime_module = None

        if aug_task == 'fill':
            train_dataset = UNetDataset(train_dataset, preprocess)
            test_dataset = UNetDataset(test_dataset, preprocess, split='test')
        elif aug_task == 'augop':   # autoaugment training
            train_dataset = UNetAugopDataset(train_dataset, preprocess)
            test_dataset = UNetAugopDataset(test_dataset, preprocess)
        elif aug_task == 'augop2':   # prime training. TODO
            pass

        return train_dataset, test_dataset, num_classes, base_c_path, prime_module


def build_dataloader(train_dataset, test_dataset, args):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    return train_loader, test_loader

