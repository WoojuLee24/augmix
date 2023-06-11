import numpy as np
from torchvision import models

import torch.backends.cudnn as cudnn

from models.allconv import AllConvNet
from models import MyDataParallel

from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from third_party.WideResNet_pytorch.wideresnet_fc import WideResNetFc
from third_party.WideResNet_pytorch.wideresnet_auxbn import WideResNetAuxBN
from third_party.WideResNet_pytorch.wideresnet_expand import WideResNetExpand
from third_party.WideResNet_pytorch.wideresnet_simsiam import WideResNetSimsiam
from third_party.WideResNet_pytorch.wideresnet_proj import WideResNetProj
from third_party.supervised_contrastive_net import SupConNet
from third_party.WideResNet_pytorch.wideresnet_encoder import WideResNetEncoder

from models.projnet import *

def build_net(args, num_classes):
    if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
        if args.model == 'densenet':
            net = densenet(num_classes=num_classes)
        elif (args.model == 'wrn') or (args.model == 'wrnexpand2'):
            net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrnfc':
            net = WideResNetFc(args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrnauxbn':
            net = WideResNetAuxBN(args, args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrnexpand':
            net = WideResNetExpand(args, args.layers, num_classes, args.widen_factor, args.droprate, args.expand_factor)
        elif args.model == 'wrnsimsiam':
            net = WideResNetSimsiam(args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrn_encoder':
            net = WideResNetEncoder(args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrnproj':
            net = WideResNetProj(args, args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'allconv':
            net = AllConvNet(num_classes)
        elif args.model == 'resnext':
            net = resnext29(num_classes=num_classes)
        else: # args.model == 'wrn':  # default == 'wrn'
            net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)

    elif args.dataset == 'imagenet100':
        print("=> creating model '{}'".format(args.model))
        net = models.__dict__[args.model]()
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif args.dataset == 'imagenet':
        # if args.pretrained:
        #     print("=> using pre-trained model '{}'".format(args.model))
        #     net = models.__dict__[args.model](pretrained=True)
        # else:
        print("=> creating model '{}'".format(args.model))
        net = models.__dict__[args.model]()

    if args.additional_loss == 'supconv0.1':
        # net = SupConNet(net, head='mlp', in_feature=2048, out_feature=128)
        net.encoder = MyDataParallel(net.encoder)
    else:
        net = MyDataParallel(net)

    net = net.cuda()
    cudnn.benchmark = True

    return net


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))

