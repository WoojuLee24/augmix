import os
import torch
from torchvision import transforms
from torchvision import datasets

from .mixdataset import BaseDataset, AugMixDataset
from .pixmix import RandomImages300K, PixMixDataset


def build_dataset(args, corrupted=False):
    dataset = args.dataset
    if dataset == 'cifar10' or 'cifar100':
        mean, std = [0.5] * 3, [0.5] * 3
    else: # imagenet
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

    if dataset == 'cifar10' or 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4)])
        test_transform = preprocess
    else: #imagenet
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip()])
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             preprocess])

    parent_dir = '/ws/data'
    dir_name = 'cifar' if (dataset == 'cifar10' or dataset == 'cifar100') else 'imagenet'
    root_dir = f'{parent_dir}/{dir_name}'

    if corrupted:
        corr1_data = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
        corr2_data = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
        return corr1_data, corr2_data
    else:
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root_dir, train=True, transform=train_transform, download=True)
            test_dataset = datasets.CIFAR10(root_dir, train=False, transform=test_transform, download=True)
            base_c_path = f'{root_dir}/CIFAR-10-C/'
            num_classes = 10
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root_dir, train=True, transform=train_transform, download=True)
            test_dataset = datasets.CIFAR100(root_dir, train=False, transform=test_transform, download=True)
            base_c_path = f'{root_dir}/CIFAR-100-C/'
            num_classes = 100
        else: # imagenet
            traindir = os.path.join(args.clean_data, 'train')
            valdir = os.path.join(args.clean_data, 'val')
            train_dataset = datasets.ImageFolder(traindir, train_transform)
            train_dataset = AugMixDataset(train_dataset, preprocess)
            test_dataset = datasets.ImageFolder(valdir, test_transform)
            base_c_path = None
            num_classes = None

        aug = args.aug
        no_jsd = args.no_jsd
        if aug == 'none':
            train_dataset = BaseDataset(train_dataset, preprocess, no_jsd)
        elif aug == 'augmix':
            train_dataset = AugMixDataset(train_dataset, preprocess, no_jsd,
                                          args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity, args.mixture_coefficient)
        elif aug == 'pixmix':
            if args.use_300k:
                mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip()]))
            else:
                mixing_set_transform = transforms.Compose([transforms.Resize(36),
                                                           transforms.RandomCrop(32)])
                mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
            train_dataset = PixMixDataset(train_dataset, mixing_set, {'normalize': normalize, 'tensorize': to_tensor},
                                          no_jsd=no_jsd, k=args.k, beta=args.beta, all_ops=args.all_ops, aug_severity=args.aug_severity)
        elif args.aug == 'arp':
            pass

        return train_dataset, test_dataset, num_classes, base_c_path


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