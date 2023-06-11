import numpy as np
from torchvision import models

import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import MyDataParallel

from third_party.UNet.unet_s import UNetS

def build_augnet(args, out_channels):
    if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
        if args.aug_model == 'unet_s':
            net = UNetS(args, out_channels=out_channels)

    elif args.dataset == 'imagenet100':
        print("=> creating model '{}'".format(args.model))
        net = models.__dict__[args.model]()
        net.fc = nn.Linear(net.fc.in_features, out_channels)

    elif args.dataset == 'imagenet':
        # if args.pretrained:
        #     print("=> using pre-trained model '{}'".format(args.model))
        #     net = models.__dict__[args.model](pretrained=True)
        # else:
        print("=> creating model '{}'".format(args.model))
        net = models.__dict__[args.model]()

    net = MyDataParallel(net)

    net = net.cuda()
    cudnn.benchmark = True

    return net


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))

