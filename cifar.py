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

import augmentations
from models.cifar.allconv import AllConvNet
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from losses import get_additional_loss
from datasets.mixdataset import BaseDataset, AugMixDataset
import wandb

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
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--additional-loss',
    '-al',
    default='jsd',
    type=str,
    choices=['none', 'jsd'],
    help='Type of additional loss')
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

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    wandb_features = {}
    total_ce_loss = 0.
    total_additional_loss = 0.
    total_correct = 0.
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.no_jsd or args.aug=='none':
            # no apply additional loss. augmentations are optional
            # aug choices = ['none', 'augmix',..]
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_ce_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
        else:
            # apply additional loss
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
            pred = logits_clean.data.max(1)[1]

            # Cross-entropy is only computed on clean images
            ce_loss = F.cross_entropy(logits_clean, targets)
            additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2)

            loss = ce_loss + additional_loss
            total_ce_loss += float(ce_loss.data)
            total_additional_loss += float(additional_loss.data)
            total_correct += pred.eq(targets.data).sum().item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        if i % args.print_freq == 0:
            print('Train Loss {:.3f}'.format(loss_ema))
    wandb_features['train/ce_loss'] = total_ce_loss / len(train_loader.dataset)
    wandb_features['train/additional_loss'] = total_additional_loss / len(train_loader.dataset)
    wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(train_loader.dataset)
    wandb_features['train/error'] = 100 - 100. * total_correct / len(train_loader.dataset)
    return loss_ema, wandb_features


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    wandb_features = dict()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    wandb_features['test/loss'] = total_loss / len(test_loader.dataset)
    wandb_features['test/error'] = 100 - 100. * total_correct / len(test_loader.dataset)
    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset), wandb_features


def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    wandb_features = dict()
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc, _ = test(net, test_loader)
        wandb_features['test_c/{}.loss'.format(corruption)] = test_loss
        wandb_features['test_c/{}.error'.format(corruption)] = 100 - 100. * test_acc
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs), wandb_features


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    if args.wandb:
        wandb.init(project='AI28', entity='ai28', name=f"{args.aug}_b{args.batch_size}")

    # Load datasets
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
        base_c_path = './data/cifar/CIFAR-100-C/'
        num_classes = 100

    # train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
    if args.aug == 'none':
        train_data = BaseDataset(train_data, preprocess, args.no_jsd)
    elif args.aug == 'augmix':
        train_data = AugMixDataset(train_data, preprocess, args.no_jsd,
                                   args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity)
    elif args.aug == 'pixmix':
        pass
        # train_data = PixMixDataset(train_data, preprocess, args.no_jsd)
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

    # Create model
    if args.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    elif args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'resnext':
        net = resnext29(num_classes=num_classes)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
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
        test_loss, test_acc, test_features = test(net, test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc, test_c_features = test_c(net, test_data, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
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

    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()

        train_loss_ema, train_features = train(net, train_loader, optimizer, scheduler)
        test_loss, test_acc, test_features = test(net, test_loader)

        # log wandb features
        for key, value in train_features.items():
            wandb.log({key: value})
        for key, value in test_features.items():
            wandb.log({key: value})

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

    test_c_acc, test_c_features = test_c(net, test_data, base_c_path)
    for key, value in test_c_features.items():
        wandb.log({key: value})
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    main()
