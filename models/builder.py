import numpy as np
from torchvision import models

import torch.backends.cudnn as cudnn

from models.allconv import AllConvNet
from models import MyDataParallel
from losses import CenterLoss, MlpJSDLoss, SupConLoss

from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from third_party.WideResNet_pytorch.wideresnetproj import WideResNetProj
from third_party.supervised_contrastive_net import SupConNet
from third_party.WideResNet_pytorch.wideresnet_encoder import WideResNetEncoder

def build_net(args, num_classes):
    if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
        if args.model == 'densenet':
            net = densenet(num_classes=num_classes)
        elif args.model == 'wrn':
            net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
        # elif args.model == 'wrn_encoder':
        #     net = WideResNetEncoder(args.layers, num_classes, args.widen_factor, args.droprate)
        elif args.model == 'wrnproj':
            net = WideResNetProj(args.layers, num_classes, args.widen_factor, args.droprate, args.jsd_layer)
        elif args.model == 'allconv':
            net = AllConvNet(num_classes)
        elif args.model == 'resnext':
            net = resnext29(num_classes=num_classes)
        else: # args.model == 'wrn':  # default == 'wrn'
            net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    else: # imagenet
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.model))
            net = models.__dict__[args.model](pretrained=True)
        else:
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


def build_loss(args, num_classes, train_loader):
    if args.additional_loss == 'center_loss':
        criterion_al = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
    elif args.additional_loss == 'mlpjsd':
        criterion_al = MlpJSDLoss(in_feature=128, out_feature=128)
    elif args.additional_loss == 'supconv0.1':
        criterion_al = SupConLoss(temperature=args.temper)
    else:
        return None

    criterion_al = criterion_al.cuda()
    return criterion_al

