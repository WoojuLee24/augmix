"""
PreTrain U-Net model
task: fill-in the deleted pixels, augmentation operators, and so on..
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np

from apis import AugTrainer, AugTester
from models import build_net, build_augnet, get_lr
from datasets import *
from datasets.aug_builder import augbuild_dataset
from utils import WandbLogger
from losses import get_ssim

from feature_hook import FeatureHook
from config import cifar10_cfg
import matplotlib.pyplot as plt


def get_args_from_parser():
    parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help='Set value to fix the seed'
                        )
    # Dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet100'],
                        help='Choose between CIFAR-10, CIFAR-100, imagenet')
    parser.add_argument('--aug', '-aug',
                        type=str,
                        default='none',
                        choices=['none', 'augmix', 'da', 'augda', 'pixmix', 'apr_s', 'prime',
                                 'augdaset', 'augdawidth'],
                        help='Choose domain generalization augmentation methods')
    parser.add_argument('--aug-task', '-augt',
                        type=str,
                        default='fill',
                        choices=['fill', 'augop'],
                        help='Choose domain generalization augmentation methods')

    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle or not')

    #####################
    ## train-aug model ##
    #####################
    # Model
    parser.add_argument('--aug-model', '-am',
                        type=str,
                        default='unet_s',
                        choices=['unet_s', 'unet_l',
                                 ],
                        help='Choose aug model architecture.')

    parser.add_argument('--aug-loss', '-agl',
                        default='msel1',
                        type=str,
                        choices=['msel1', 'mse', 'l1', 'ssim', 'msel1ssim',
                                 ],
                        help='Type of additional loss')

    parser.add_argument('--alw', '-alw', default=0.1, type=float, help='aug additional loss weight')
    parser.add_argument('--alw2', '-alw2', default=0.5, type=float, help='aug additional loss weight')
    parser.add_argument('--alw3', '-alw3', default=1.0, type=float, help='aug additional loss weight')



    # ## AugMix options
    # parser.add_argument('--mixture-width', default=3, type=int,
    #                     help='Number of augmentation chains to mix per augmented example')
    # parser.add_argument('--mixture-depth', default=-1, type=int,
    #                     help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    # parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
    # parser.add_argument('--mixture-coefficient', '-mc', default=1.0, type=float, help='mixture coefficient alpha')
    # parser.add_argument('--mixture-alpha', '-ma', default=1.0, type=float, help='mixture coefficient alpha')
    # parser.add_argument('--mixture-beta', '-mb', default=1.0, type=float, help='mixture coefficient beta')
    # parser.add_argument('--da-prob', default=1/3, type=float,
    #                     help='deepaugment probability for augdaset mode')
    #
    # parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
    # parser.add_argument('--additional-loss', '-al',
    #                     default='jsd',
    #                     type=str,
    #                     choices=['none', 'jsd',
    #                              'kl',
    #                              'klv1.0', 'klv1.1', 'klv1.2', 'klv1.3',
    #                              'msev1.0',
    #                              ],
    #                     help='Type of additional loss')
    # parser.add_argument('--reduction',
    #                     default='batchmean',
    #                     type=str,
    #                     choices=['batchmean', 'none', 'mean', 'sum',
    #                              ],
    #                     help='additional loss mean')
    # parser.add_argument('--temper', default=1.0, type=float, help='temperature scaling')
    # parser.add_argument('--lambda-weight', '-lw', default=12.0, type=float, help='additional loss weight')
    # parser.add_argument('--skew', default=0.8, type=float, help='skew parameter for logit')
    #
    # # feature loss
    # parser.add_argument('--additional-loss2', '-al2',
    #                     default='jsd',
    #                     type=str,
    #                     choices=['none', 'jsd',
    #                              'kl',
    #                              'klv1.0', 'klv1.1', 'klv1.2', 'klv1.3',
    #                              'msev1.0',
    #                              'csl2',
    #                              'ssim',
    #                              ],
    #                     help='Type of additional loss')
    #
    ## ssim option ##
    parser.add_argument('--window', '-w', default=3, type=int, help='window size for gaussian')
    parser.add_argument('--sigma', '-sigma0', default=0.5, type=float, help='sigma size for gaussian')

    #
    # ## hook option ##
    # parser.add_argument('--hook', action='store_true', help='hook layers for feature extraction')
    # parser.add_argument('--hook-layer', default='None', help='which layer to hook')
    # parser.add_argument('--all-ops', '-all', action='store_true',
    #                     help='Turn on all operations (+brightness,contrast,color,sharpness).')
    #
    # ## PIXMIX options
    # parser.add_argument('--beta', default=3, type=int, help='Severity of mixing')
    # parser.add_argument('--k', default=4, type=int, help='Mixing iterations')
    # parser.add_argument('--mixing-set', type=str, default='/ws/data/fractals_and_fvis/', help='Mixing set directory.')
    # parser.add_argument('--use_300k', action='store_true', help='use 300K random images as aug data')
    #
    # ## APR options
    # parser.add_argument('--apr_p', action='store_true', help='recommend to do apr_p when using apr-s' )
    # parser.add_argument('--apr_mixed_coefficient', '-aprmc', default=0.5, type=float, help='probability of using apr-p and apr-s')
    #
    # # Model
    # parser.add_argument('--model', '-m',
    #                     type=str,
    #                     default='wrn',
    #                     choices=['wrn', 'allconv', 'densenet', 'resnext',
    #                              ### imagenet ###
    #                              'resnet50'],
    #                     help='Choose architecture.')
    #
    # ## WRN Architecture options
    # parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    # parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
    # parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--optim', type=str, default='sgd', help='which optimizer to use')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size.')
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

    # Checkpointing options
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--save-every', '-se', type=bool, default=False, help='save checkpoints every time.')
    parser.add_argument('--resume', '-r', type=str, default='', help='Checkpoint path for resume / test.')
    parser.add_argument('--print-freq', type=int, default=50, help='Training loss print frequency (batches).')

    # Acceleration
    parser.add_argument('--num-workers', type=int, default=4, help='Number of pre-fetching threads.')

    # Log
    parser.add_argument('--evaluate', action='store_true', help='Eval only.')
    parser.add_argument('--analysis', action='store_true', help='Analysis only. ')
    parser.add_argument('--log-freq', type=int, default=100, help='Training log frequency (batches) in wandb.')
    parser.add_argument('--wandb', '-wb', action='store_true', help='Turn on wandb log')
    parser.add_argument('--confusion-matrix', '-cm', action='store_true', help='Turn on wandb log')
    parser.add_argument('--debug', action='store_true', help='Debugging')

    args = parser.parse_args()

    return args


def imsave(image, root, name):
    root = os.path.join(root, 'img')
    os.makedirs(root, exist_ok=True)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.cpu().detach().numpy()
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    plt.imsave(root + f'/{name}.png', np.asarray(image))


def main():
    args = get_args_from_parser()
    if args.seed == -1:
        torch.manual_seed(1)
        np.random.seed(1)
    else:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        # args.num_workers = 0
        random.seed(seed)

    ########################
    ### Initialize wandb ###
    ########################
    if args.evaluate:
        resume_path = (args.resume).split('/')
        resume_model = resume_path[-2]
        name = resume_model + '_evaluate'

    else:
        save_path = args.save
        name = save_path.split("/")[-1]

    if args.wandb:
        wandg_config = dict(project="hendrycks", entity='kaist-url-ai28', name=name,)#  resume=args.resume)
        wandb_logger = WandbLogger(wandg_config, args)
    else:
        wandb_logger = WandbLogger(None)
    wandb_logger.before_run()

    #####################
    ####### prime ######
    #####################

    if args.aug == 'prime':
        if args.dataset == 'cifar10':
            args.prime = cifar10_cfg.get_config()

    #####################
    ### Load datasets ###
    #####################

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagprime_moduleenet':
        args.num_classes = 1000

    train_dataset, test_dataset, num_classes, base_c_path, prime_module = augbuild_dataset(args)
    train_loader, test_loader = build_dataloader(train_dataset, test_dataset, args)

    ####################
    ### Create model ###
    ####################
    if args.aug_task == 'fill':
        out_channels = 3
    elif args.aug_task == 'augop':
        out_channels = 30 # TODO
    aug_net = build_augnet(args, out_channels)

    #################
    ### Optimizer ###
    #################
    # TODO: what effect?
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            aug_net.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            aug_net.parameters(),
            args.learning_rate)


    ##############
    ### Resume ###
    ##############
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            aug_net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print('Aug model restored from epoch:', start_epoch)

    #################
    ### Scheduler ###
    #################
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))


    ## wandb ##
    if args.wandb:
        wandg_config = dict(project="hendrycks", entity='kaist-url-ai28', name=name,)#  resume=args.resume)
        wandb_logger = WandbLogger(wandg_config, args)
    else:
        wandb_logger = WandbLogger(None)
    wandb_logger.before_run()

    aug_net = aug_net.to('cuda')
    # train
    for epoch in range(start_epoch, args.epochs):
        trainer = AugTrainer(aug_net, args, optimizer, scheduler)
        tester = AugTester(aug_net, args)
        save_img = True if epoch == args.epochs - 1 else False

        if args.aug_task == 'fill':
            train_features = trainer.train(train_loader)
            test_features = tester.test(test_loader, save_img)
        elif args.aug_task == 'augop':
            train_features = trainer.train_augop(train_loader)
            test_features = tester.test_augop(test_loader, save_img)

        wandb_logger.log_evaluate(dict(train_features=train_features,
                                       test_features=test_features))

    # _ = tester.test(test_loader, save_img=True)

    # save pth and log files: TODO
    checkpoint = {
        'epoch': epoch,
        'aug_task': args.aug_task,
        'model': args.aug_model,
        'state_dict': aug_net.state_dict(),
        'ssim': test_features['test/ssim'],
        'l1_loss': test_features['test/l1_loss'],
        'mse_loss': test_features['test/mse_loss'],
        'optimizer': optimizer.state_dict(),
    }
    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(args.save, f"checkpoint{epoch}.pth.tar")
    torch.save(checkpoint, save_path)


if __name__ == '__main__':
    # config_flags.DEFINE_config_file('config') # prime
    main()