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
from datasets import *
from losses import get_additional_loss, CenterLoss
from datasets.mixdataset import BaseDataset, AugMixDataset
from datasets.concatdataset import ConcatDataset
from feature_hook import FeatureHook
from utils import plot_confusion_matrix
from utils import plot_tsne
import pandas as pd
import wandb
import random

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
    choices=['none', 'jsd', 'jsd_temper', 'kl', 'ntxent', 'center_loss'],
    help='Type of additional loss')
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
            additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
                                                  12, targets)

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


def train2(net, train_loader, criterion_al, optimizer, optimizer_al, scheduler):
    """Train for one epoch."""
    net.train()
    wandb_features = {}
    total_ce_loss = 0.
    total_additional_loss = 0.
    total_correct = 0.
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_al.zero_grad()

        if args.no_jsd or args.aug == 'none':
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
            additional_loss = criterion_al(net.module.features, targets)
            # additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
            #                                       12, targets)

            loss = ce_loss + additional_loss
            total_ce_loss += float(ce_loss.data)
            total_additional_loss += float(additional_loss.data)
            total_correct += pred.eq(targets.data).sum().item()

        loss.backward()
        optimizer.step()
        # by doing so, weight_cent would not impact on the learning of centers\
        weight_cent = 1
        for param in criterion_al.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_al.step()
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
    confusion_matrix = torch.zeros(10, 10)
    tsne_features = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
            # plt = plot_tsne(net.module.features, targets)
            # plt.savefig("/ws/data/log/debug.jpg")
            # tsne_features.append(net.module.features)
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    wandb_features['test/loss'] = total_loss / len(test_loader.dataset)
    wandb_features['test/error'] = 100 - 100. * total_correct / len(test_loader.dataset)
    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset), wandb_features, confusion_matrix


def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    wandb_features = dict()
    wandb_plts = dict()
    wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
    confusion_matrices = []
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

        test_loss, test_acc, _, confusion_matrix = test(net, test_loader)
        # wandb_features['test_c/{}.loss'.format(corruption)] = test_loss
        # wandb_features['test_c/{}.error'.format(corruption)] = 100 - 100. * test_acc
        wandb_table[corruption]['loss'] = test_loss
        wandb_table[corruption]['error'] = 100 - 100. * test_acc
        # wandb_plts[corruption] = confusion_matrix
        corruption_accs.append(test_acc)
        confusion_matrices.append(confusion_matrix.cpu().detach().numpy())
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    # return np.mean(corruption_accs), wandb_features
    return  np.mean(corruption_accs), wandb_table, np.mean(confusion_matrices, axis=0)


def test_c_dg(net, test_data, corr1_data, corr2_data, base_path):
    """
    Evaluate additional loss on given combinations of corrupted datasets.
    Each corrupted dataset are compared with the same level corrupted dataset.
    """
    wandb_features = dict()
    total_additional_loss = 0.
    from itertools import combinations
    wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=CORRUPTIONS)
    for corruption1, corruption2 in combinations(CORRUPTIONS, 2):
        clean_loss = 0.
        corr_additional_loss = 0.
        test_data.data = np.load(base_path + 'clean.npy')
        corr1_data.data = np.load(base_path + corruption1 + '.npy')
        corr2_data.data = np.load(base_path + corruption2 + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        corr1_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        corr2_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        concat_data = ConcatDataset((test_data, corr1_data, corr2_data))

        test_loader = torch.utils.data.DataLoader(
            concat_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)
        # test_loss, test_acc, _ = test(net, test_loader)
        with torch.no_grad():
            for clean, corr1, corr2 in test_loader:
                images = torch.cat([clean[0], corr1[0], corr2[0]], dim=0)
                targets = torch.cat([clean[1], corr1[1], corr2[1]], dim=0)
                images, targets = images.cuda(), targets.cuda()
                logits = net(images)
                logits_clean, logits_aug1, logits_aug2 = torch.chunk(logits, 3)
                target_clean, target_aug1, target_aug2 = torch.chunk(targets, 3)
                loss = F.cross_entropy(logits_clean, target_clean)
                additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2)
                clean_loss += float(loss.data)
                corr_additional_loss += float(additional_loss.data)
            # wandb_features['test_c/loss_clean'] = clean_loss / len(test_loader)
            # wandb_features['test_c/additional_loss_{}_{}'.format(corruption1, corruption2)] = \
            #     corr_additional_loss / len(test_loader)
            wandb_table[corruption1][corruption2] = corr_additional_loss / len(test_loader)
            print('test_c/loss_clean: ', clean_loss / len(test_loader))
            print('test_c/additional_loss_{}_{}'.format(corruption1, corruption2), corr_additional_loss / len(test_loader))

        total_additional_loss += corr_additional_loss
    combinations_length = sum(1 for _ in combinations(CORRUPTIONS, 2))
    # wandb_features['test_c/additional_loss_total'.format(total_additional_loss)] = \
    #     total_additional_loss / combinations_length
    print('test_c/additional_loss_total'.format(total_additional_loss), total_additional_loss / combinations_length)

    # wandb_table = wandb.Table(data=df)

    return total_additional_loss / combinations_length, wandb_features, wandb_table


def main():
    torch.manual_seed(1)
    np.random.seed(1)
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
        wandb.init(project='AI28', entity='kaist-url-ai28', name=name)

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
        base_c_path = '/ws/data/cifar/CIFAR-100-C/'
        num_classes = 100

    # train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
    if args.aug == 'none':
        train_data = BaseDataset(train_data, preprocess, args.no_jsd)
    elif args.aug == 'augmix':
        train_data = AugMixDataset(train_data, preprocess, args.no_jsd,
                                   args.all_ops, args.mixture_width, args.mixture_depth, args.aug_severity)
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

    # Create model
    if args.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    elif args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'resnext':
        net = resnext29(num_classes=num_classes)

    # Create additional loss model
    if args.additional_loss == 'center_loss':
        criterion_al = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
        optimizer_al = torch.optim.SGD(criterion_al.parameters(), lr=0.5)

    # Hook Layers
    if args.hook:
        hook = FeatureHook(["block3.layer.5.conv2"])
        hook.hook_multi_layer(net)

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
        test_loss, test_acc, test_features, test_cm = test(net, test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc, test_c_table, test_c_cm = test_c(net, test_data, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

        # log wandb features
        for key, value in test_features.items():
            wandb.log({key: value})
        test_plt = plot_confusion_matrix(test_cm)
        wandb.log({'clean': test_plt})
        test_c_table = wandb.Table(data=test_c_table)
        wandb.log({"test_c_results": test_c_table})
        wandb.log({"test/corruption_error: ": 100 - 100. * test_c_acc})
        test_c_plt = plot_confusion_matrix(test_c_cm)
        wandb.log({'corruption': test_c_plt})
        # for key, value in test_c_features.items():
        #     wandb.log({key: value})
        # test_c_plt = plot_confusion_matrix(test_c_cm)
        # for key, value in test_c_plt.items():
        #     wandb.log({key: value})

        return

    elif args.analysis:
        corr1_data = datasets.CIFAR10(
            '/ws/data/cifar', train=False, transform=test_transform, download=True)
        corr2_data = datasets.CIFAR10(
            '/ws/data/cifar', train=False, transform=test_transform, download=True)
        # test_c_dg
        test_dg_loss, test_dg_features, test_dg_table = test_c_dg(net, test_data, corr1_data, corr2_data, base_c_path)
        test_dg_table = wandb.Table(data=test_dg_table)
        wandb.log({"additional_loss": test_dg_table})
        for key, value in test_dg_features.items():
            wandb.log({key: value})

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
        if args.additional_loss in ['center_loss']:
            train_loss_ema, train_features = train2(net, train_loader, criterion_al, optimizer, optimizer_al, scheduler)
        else:
            train_loss_ema, train_features = train(net, train_loader, optimizer, scheduler)
        test_loss, test_acc, test_features, test_cm = test(net, test_loader)

        # log wandb features
        if args.wandb:
            for key, value in train_features.items():
                wandb.log({key: value})
            for key, value in test_features.items():
                wandb.log({key: value})
            test_plt = plot_confusion_matrix(test_cm)
            wandb.log({'clean': test_plt})

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

    test_c_acc, test_c_table, test_c_cm = test_c(net, test_data, base_c_path)
    if args.wandb:
        test_c_table = wandb.Table(data=test_c_table)
        wandb.log({"test_c_results": test_c_table})
        # for key, value in test_c_features.items():
        #     wandb.log({key: value})

    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    wandb.log({"test/corruption_error: ": 100 - 100. * test_c_acc})
    test_c_plt = plot_confusion_matrix(test_c_cm)
    wandb.log({'clean': test_c_plt})

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    main()
