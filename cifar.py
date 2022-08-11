# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

from models.cifar.allconv import AllConvNet
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from datasets import *
from losses import CenterLoss, MlpJSDLoss
from datasets.mixdataset import BaseDataset, AugMixDataset
from feature_hook import FeatureHook
from utils import WandbLogger
from apis import test, test_c, test_c_dg, Trainer


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))
def get_args_from_parser():
    parser = argparse.ArgumentParser(
        description='Trains a CIFAR Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100'],
        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument(
        '--aug',
        '-aug',
        type=str,
        default='augmix',
        choices=['none', 'augmix', 'pixmix', 'apr'],
        help='Choose domain generalization augmentation methods')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='wrn',
        choices=['wrn', 'allconv', 'densenet', 'resnext'],
        help='Choose architecture.')
    # Optimization options
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument(
        '--learning-rate',
        '-lr',
        type=float,
        default=0.1,
        help='Initial learning rate.')
    parser.add_argument(
        '--batch-size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument(
        '--decay',
        '-wd',
        type=float,
        default=0.0005,
        help='Weight decay (L2 penalty).')
    # WRN Architecture options
    parser.add_argument(
        '--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
    parser.add_argument(
        '--droprate', default=0.0, type=float, help='Dropout probability')

    # AugMix options
    parser.add_argument(
        '--mixture-width',
        default=3,
        type=int,
        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument(
        '--mixture-depth',
        default=-1,
        type=int,
        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument(
        '--aug-severity',
        default=3,
        type=int,
        help='Severity of base augmentation operators')
    parser.add_argument(
        '--mixture-coefficient',
        '-mc',
        default=1.0,
        type=float,
        help='mixture coefficient alpha')
    parser.add_argument(
        '--no-jsd',
        '-nj',
        action='store_true',
        help='Turn off JSD consistency loss.')
    parser.add_argument(
        '--additional-loss',
        '-al',
        default='jsd',
        type=str,
        choices=['none', 'jsd', 'jsd_temper', 'kl', 'ntxent', 'center_loss', 'mlpjsd'],
        help='Type of additional loss')
    parser.add_argument(
        '--temper',
        default=1.0,
        type=float,
        help='temperature scaling')
    parser.add_argument(
        '--reduction',
        default='batchmean',
        type=str,
        choices=['batchmean', 'mean'],
        help='temperature scaling')
    parser.add_argument(
        '--jsd-layer',
        default='features',
        type=str,
        choices=['features', 'logits'],
        help='apply jsd loss for the selected layer')
    parser.add_argument(
        '--hook',
        action='store_true',
        help='hook layers for feature extraction')
    parser.add_argument(
        '--all-ops',
        '-all',
        action='store_true',
        help='Turn on all operations (+brightness,contrast,color,sharpness).')
    # Checkpointing options
    parser.add_argument(
        '--save',
        '-s',
        type=str,
        default='/ws/data/log',
        help='Folder to save checkpoints.')
    parser.add_argument(
        '--resume',
        '-r',
        type=str,
        default='',
        help='Checkpoint path for resume / test.')
    parser.add_argument('--evaluate', action='store_true', help='Eval only.')
    parser.add_argument('--analysis', action='store_true', help='Analysis only. ')
    parser.add_argument(
        '--print-freq',
        type=int,
        default=50,
        help='Training loss print frequency (batches).')
    # Acceleration
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of pre-fetching threads.')
    parser.add_argument(
        '--wandb',
        '-wb',
        action='store_true',
        help='Turn on wandb log')
    parser.add_argument(
        '--confusion-matrix',
        '-cm',
        action='store_true',
        help='Turn on wandb log')

    ### PIXMIX
    parser.add_argument(
        '--beta',
        default=3,
        type=int,
        help='Severity of mixing')
    parser.add_argument(
        '--k',
        default=4,
        type=int,
        help='Mixing iterations')
    parser.add_argument(
        '--mixing-set',
        type=str,
        default='/ws/data/fractals_and_fvis/',
        help='Mixing set directory.')
    parser.add_argument(
        '--use_300k',
        action='store_true',
        help='use 300K random images as aug data'
    )

    args = parser.parse_args()

    return args

args = get_args_from_parser()

def main():
    torch.manual_seed(1)
    np.random.seed(1)

    ''' Initialize wandb '''
    if args.evaluate:
        resume_path = (args.resume).split('/')
        resume_model = resume_path[-2]
        name = resume_model + '_evaluate'
    elif args.analysis:
        resume_path = (args.resume).split('/')
        resume_model = resume_path[-2]
        name = resume_model + '_analysis'
    else:
        name = f"{args.aug}_{args.additional_loss}_b{args.batch_size}"
    if args.wandb:
        wandg_config = dict(project='AI28', entity='kaist-url-ai28', name=name)
        wandb_logger = WandbLogger(wandg_config, args)
    else:
        wandb_logger = WandbLogger(None)
    wandb_logger.before_run() # wandb here

    ''' Load datasets '''
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            '/ws/data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            '/ws/data/cifar', train=False, transform=test_transform, download=True)
        base_c_path = '/ws/data/cifar/CIFAR-10-C/'
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        base_c_path = '/ws/data/cifar/CIFAR-100-C/'
        num_classes = 100

    # train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
    if args.aug == 'none':
        train_data = BaseDataset(train_data, preprocess, args.no_jsd)
    elif args.aug == 'augmix':
        train_data = AugMixDataset(train_data, preprocess, args.no_jsd,
                                   args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity, args.mixture_coefficient)
    elif args.aug == 'pixmix':
        if args.use_300k:
            mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip()]))
        else:
            mixing_set_transform = transforms.Compose(
                [transforms.Resize(36),
                 transforms.RandomCrop(32)])
            mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
        train_data = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor},
                                   no_jsd=args.no_jsd, k=args.k, beta=args.beta, all_ops=args.all_ops, aug_severity=args.aug_severity)
    elif args.aug == 'arp':
        pass
        # train_data = PixMixDataset(train_data, preprocess, args.no_jsd)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    ''' Create model '''
    if args.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    elif args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'resnext':
        net = resnext29(num_classes=num_classes)
    else: # default == 'wrn'
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)

    ''' Create additional loss model '''
    if args.additional_loss == 'center_loss':
        criterion_al = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
        optimizer_al = torch.optim.SGD(criterion_al.parameters(), lr=0.5)
    elif args.additional_loss == 'mlpjsd':
        criterion_al = MlpJSDLoss(in_feature=128, out_feature=128)
        criterion_al = criterion_al.cuda()
        optimizer_al = torch.optim.SGD(criterion_al.parameters(), lr=args.learning_rate)
        scheduler_al = torch.optim.lr_scheduler.LambdaLR(
            optimizer_al,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(train_loader),
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))

    ''' Hook Layers '''
    if args.hook:
        hook = FeatureHook(["block1.layer.5.conv2",
                            "block2.layer.5.conv2",
                            "block3.layer.5.conv2"
                            ])
        hook.hook_multi_layer(net)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    # Distribute model across all visible GPUs
    from models import MyDataParallel
    # net = torch.nn.DataParallel(net).cuda()
    net = MyDataParallel(net).cuda()
    cudnn.benchmark = True

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)

    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc, test_features, test_cm = test(net, test_loader, args)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc, test_c_table, test_c_cm = test_c(net, test_data, args, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

        # wandb here
        wandb_logger.log_evaluate(dict(test_features=test_features,
                                       test_cm=test_cm,
                                       test_c_table=test_c_table,
                                       test_c_acc=test_c_acc,
                                       test_c_cm=test_c_cm))

        return

    elif args.analysis:
        corr1_data = datasets.CIFAR10(
            '/ws/data/cifar', train=False, transform=test_transform, download=True)
        corr2_data = datasets.CIFAR10(
            '/ws/data/cifar', train=False, transform=test_transform, download=True)
        # test_c_dg
        test_dg_loss, test_dg_features, test_dg_table = test_c_dg(net, test_data, args, corr1_data, corr2_data, base_c_path)

        # wandb here
        wandb_logger.log_analysis(dict(test_dg_table=test_dg_table,
                                       test_dg_features=test_dg_features))

        return

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    log_path = os.path.join(args.save,
                            args.dataset + '_' + args.model + '_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    trainer = Trainer(net, wandb_logger=wandb_logger)
    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        wandb_logger.before_train_epoch() # wandb here
        begin_time = time.time()
        if args.additional_loss in ['center_loss', 'mlpjsd']:
            train_loss_ema, train_features = trainer.train2(train_loader, args,  optimizer, scheduler,
                                                            criterion_al, optimizer_al, scheduler_al)
        else:
            train_loss_ema, train_features = trainer.train(train_loader, args, optimizer, scheduler)
        wandb_logger.after_train_epoch(dict(train_features=train_features)) # wandb here

        test_loss, test_acc, test_features, test_cm = test(net, test_loader, args)
        wandb_logger.after_test_epoch(dict(test_features=test_features, # wandb here
                                            test_cm=test_cm))

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.model,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(args.save, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                train_loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}'
            .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                    test_loss, 100 - 100. * test_acc))

    test_c_acc, test_c_table, test_c_cm = test_c(net, test_data, args, base_c_path)
    wandb_logger.after_run(dict(test_c_table=test_c_table, # wandb here
                                test_c_acc=test_c_acc,
                                test_c_cm=test_c_cm))

    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    main()