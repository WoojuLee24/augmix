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
import torch
import random
import torch.backends.cudnn as cudnn

import numpy as np

from apis import Trainer, Tester
from models import build_net, build_loss, AdditionalLoss, get_lr
from datasets import *
from utils import WandbLogger

from feature_hook import FeatureHook
from config import cifar10_cfg


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
                        default='augmix',
                        choices=['none', 'augmix', 'da', 'augda', 'pixmix', 'augmixv2', 'apr_s', 'prime', 'ctrlaugmix',
                                 'augdaset', 'augdawidth'],
                        help='Choose domain generalization augmentation methods')
    parser.add_argument('--aux-aug', '-auxa',
                        type=str,
                        default='none',
                        choices=['none', 'da', 'daall', 'augmix', 'pixmix', 'augmixv2', 'apr_s', 'prime', 'ctrlaugmix'],
                        help='Choose aux domain generalization augmentation methods')

    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle or not')

    ## AugMix options
    parser.add_argument('--mixture-width', default=3, type=int,
                        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument('--mixture-depth', default=-1, type=int,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
    parser.add_argument('--mixture-coefficient', '-mc', default=1.0, type=float, help='mixture coefficient alpha')
    parser.add_argument('--mixture-alpha', '-ma', default=1.0, type=float, help='mixture coefficient alpha')
    parser.add_argument('--mixture-beta', '-mb', default=1.0, type=float, help='mixture coefficient beta')
    parser.add_argument('--mixture-fix', '-mf', action='store_true', help='mixture coefficient fix: use mixture-alpha as coefficient')
    parser.add_argument('--da-prob', default=1/3, type=float,
                        help='deepaugment probability for augdaset mode')

    parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
    parser.add_argument('--additional-loss', '-al',
                        default='jsd',
                        type=str,
                        choices=['none', 'jsd', 'jsd.manual', 'jsd.manual.ce','jsd_temper', 'pjsd',
                                 'jsdvl_v0.1', 'jsdvl_v0.1.1', 'jsdvl_v0.1.2',
                                 'jsd.skew',
                                 'analysisv1.0',
                                 'jsdv1',
                                 'jsdv2', 'jsdv2.1',
                                 'jsdv3', 'jsdv3.0.1', 'jsdv3.0.2', 'jsdv3.0.3', 'jsdv3.0.4',
                                 'jsdv3.0.1.detach', 'jsdv3.0.2.detach', 'jsdv3.0.3.detach',
                                 'jsdv3.test', 'jsdv3.1', 'jsdv3.1.1', 'jsdv3.log.inv', 'jsdv3.inv', 'jsdv3.msev1.0', 'jsdv3.msev1.1', 'jsdv3.msev1.0.detach',
                                 'jsdv3.cossim', 'jsdv3.simsiam', 'jsdv3.simsiamv0.1',
                                 'jsdv4',
                                 'jsd.ntxent', 'jsd.ntxentv0.01', 'jsd.ntxentv0.02',
                                 'jsdv3.ntxent', 'jsdv3.ntxentv0.01', 'jsdv3.ntxentv0.02', 'jsdv3.ntxent.detach',
                                 'kl',
                                 'supconv0.01', 'supconv0.01_test', 'supconv0.01.diff',
                                 'ntxent', 'center_loss', 'mlpjsd', 'mlpjsdv1.1', 'jsdv3_apr_p',
                                 'nojsd_apr_p',
                                 'klv1.0', 'klv1.1', 'klv1.2', 'klv1.3',
                                 'klv1.0.detach', 'klv1.1.detach', 'klv1.2.detach',
                                 'klv1.0.inv', 'klv1.1.inv', 'klv1.2.inv',
                                 'msev1.0', 'msev1.0.detach',
                                 ],
                        help='Type of additional loss')
    parser.add_argument('--temper', default=1.0, type=float, help='temperature scaling')
    parser.add_argument('--temper2', default=1.0, type=float, help='temperature scaling')
    parser.add_argument('--lambda-weight', '-lw', default=12.0, type=float, help='additional loss weight')
    parser.add_argument('--lambda-alpha', '-la', default=0.0, type=float, help='additional loss weight alpha')
    parser.add_argument('--lambda-beta', '-lb', default=1.0, type=float, help='additional loss weight beta')
    parser.add_argument('--prob', default=0.1, type=float, help='prob for pjsd')

    parser.add_argument('--additional-loss2', '-al2',
                        default='none',
                        type=str,
                        choices=['none', 'jsd',
                                 'jsdvl_v0.1',
                                 'msev1.0', 'msev1.1',
                                 'jsdv4.ntxent', 'jsdv4.ntxentv0.01', 'jsdv4.ntxentv0.02', 'jsdv4.ntxent.detach',
                                 'opl',
                                 'cossim',
                                 'csl2',
                                 'cslp',
                                 'cslp_mean',
                                 'cslp_jsd',
                                 'cslp_ce',
                                 'cslp_jsd_ce',
                                 'aux_jsd',
                                 'cssoftmax',
                                 'cslpsoftmax',
                                 'ssim',
                                 'ssim_multi',
                                 'njsd',
                                 ],
                        help='Type of additiona loss2')

    parser.add_argument('--lambda-weight2', '-lw2', default=12.0, type=float, help='additional loss weight2')
    parser.add_argument('--lambda-weight3', '-lw3', default=12.0, type=float, help='additional loss weight2')

    parser.add_argument('--skew', default=0.8, type=float, help='skew parameter for logit')


    # uniform-label ##
    parser.add_argument('--uniform-label', '-ul',
                        type=str,
                        default='none',
                        choices=['none', 'v0.1', 'v0.2', 'v0.3'],
                        help='Choose domain generalization augmentation methods')
    parser.add_argument('--aux-label',
                        type=str,
                        default='target',
                        choices=['none', 'uniform', 'target', 'v0.1', 'v0.2', 'v0.3'],
                        help='aux dataset label')
    parser.add_argument('--aux-dataset', '-auxd',
                        type=str,
                        default='none',
                        choices=['none', 'fractals', 'imagenet', 'mmix', 'cifar10'],
                        help='Choose auxiliary datasets')
    parser.add_argument('--aux-num', '-auxn',
                        type=int,
                        default=1,
                        help='number of images with uniform labels')
    parser.add_argument('--aux-prob', '-auxp',
                        type=float,
                        default=0.125,
                        help='prop of aux augmentation applied')
    parser.add_argument('--aux-sample', '-auxsp',
                        type=str,
                        default='none',
                        choices=['none', 'balanced'],
                        help='prop of aux augmentation applied')
    parser.add_argument('--aux-type', '-auxt',  # will be deprecated
                        type=str,
                        default='none',
                        choices=['none', 'unoise', 'gnoise', 'mix', 'mix_unoise', 'mixup_unoise', 'fractals'],
                        help='number of images with uniform labels')
    parser.add_argument('--aux-severity', '-auxs',  # will be deprecated
                        type=float,
                        default=1,
                        help='severity of uniform noise,..')
    parser.add_argument('--aux-lambda', '-auxl',
                        type=float,
                        default=1,
                        help='lambda of uniform label loss')
    parser.add_argument('--aux-lambda2', '-auxl2',
                        type=float,
                        default=1,
                        help='lambda of additional loss')
    parser.add_argument('--aux-hlambda', '-auxhl',
                        type=float,
                        default=1,
                        help='lambda of uniform label loss')
    parser.add_argument('--mmix-severity', '-mmixs',
                        type=int,
                        default=16,
                        help='mmix number of images')

    ## ssim option ##
    parser.add_argument('--window', '-w', default=3, type=int, help='window size for gaussian')

    ## opl option ##
    parser.add_argument('--opl-norm', action='store_true', help='opl feature normalization')
    parser.add_argument('--opl-attention', action='store_true', help='opl feature attention')
    parser.add_argument('--opl-gamma', default=2, type=float, help='opl gamma parameter')

    parser.add_argument('--reduction', default='batchmean', type=str, choices=['batchmean', 'mean'],
                        help='temperature scaling')
    parser.add_argument('--margin', default=0.02, type=float, help='triplet loss margin')
    parser.add_argument('--jsd-layer', default='features', type=str, choices=['features', 'logits'],
                        help='apply jsd loss for the selected layer')
    parser.add_argument('--hook', action='store_true', help='hook layers for feature extraction')
    parser.add_argument('--hook-layer', default='None', help='which layer to hook')
    parser.add_argument('--all-ops', '-all', action='store_true',
                        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    ## PIXMIX options
    parser.add_argument('--beta', default=3, type=int, help='Severity of mixing')
    parser.add_argument('--k', default=4, type=int, help='Mixing iterations')
    parser.add_argument('--mixing-set', type=str, default='/ws/data/fractals_and_fvis/', help='Mixing set directory.')
    parser.add_argument('--use_300k', action='store_true', help='use 300K random images as aug data')

    ## APR options
    parser.add_argument('--apr_p', action='store_true', help='recommend to do apr_p when using apr-s' )
    parser.add_argument('--apr_mixed_coefficient', '-aprmc', default=0.5, type=float, help='probability of using apr-p and apr-s')

    # Model
    parser.add_argument('--model', '-m',
                        type=str,
                        default='wrn',
                        choices=['wrn', 'wrnfc', 'wrnauxbn', 'wrnexpand', 'wrnexpand2', 'wrnproj', 'wrnsimsiam', 'allconv', 'densenet', 'resnext',
                                 ### imagenet ###
                                 'resnet50'],
                        help='Choose architecture.')
    ### Siamese Architecture options
    parser.add_argument('--siamese', '-siam', action='store_true', help='basic siamese network architecture')
    ## WRN Architecture options
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')

    ## WRNExpand Architecture options
    parser.add_argument('--expand-factor', default=2, type=int, help='Expand factor')

    ## WRNAuxBN Architecture options
    parser.add_argument('--aux', default=3, type=str, help='AuxBN factor')

    ## WRNProj Architecture options
    parser.add_argument('--proj-from', default='feature',  choices=['feature', 'logit'], type=str, help='proj from which layer')
    parser.add_argument('--num-proj', default=2, type=int, help='number of proj fc layers')
    parser.add_argument('--proj-in', default=128, type=int, help='channels of proj fc layers')
    parser.add_argument('--proj-out', default=128, type=int, help='channels of proj fc layers')

    ## WRNFc Architecture options
    parser.add_argument('--fcnoise', default='none', type=str, help='fc noise type: none, unoise, gnoise,..')
    parser.add_argument('--fcnoise-s', default=0.1, type=float, help='fc nosise severity')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
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

    # save grad analysis options
    parser.add_argument('--save-grad', '-sg', action='store_true', help='save gradient ')
    parser.add_argument('--set-ops', '-sop', type=int, default=-1, help='select which operation to use')

    # cls dg training
    parser.add_argument('--cls-dg', '-cd', type=int, default=-1, help='which cls dg')

    # Log
    parser.add_argument('--evaluate', action='store_true', help='Eval only.')
    parser.add_argument('--analysis', action='store_true', help='Analysis only. ')
    parser.add_argument('--log-freq', type=int, default=100, help='Training log frequency (batches) in wandb.')
    parser.add_argument('--wandb', '-wb', action='store_true', help='Turn on wandb log')
    parser.add_argument('--confusion-matrix', '-cm', action='store_true', help='Turn on wandb log')
    parser.add_argument('--debug', action='store_true', help='Debugging')

    args = parser.parse_args()

    return args


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
    elif args.analysis:
        resume_path = (args.resume).split('/')
        resume_model = resume_path[-2]
        name = resume_model + '_analysis'
        save_path = os.path.join(args.save, "analysis")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    elif args.save_grad:
        args.shuffle = False
        if not args.resume=='':
            resume_path = (args.resume).split('/')
            resume_model = resume_path[-2]
            name = resume_model + '_sg'
            save_path = os.path.join(args.save, "sg")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
    else:
        # name = f"{args.aug}_{args.additional_loss}_b{args.batch_size}"
        save_path = args.save
        name = save_path.split("/")[-1]

    if args.wandb:
        wandg_config = dict(project="Classification", entity='kaist-url-ai28', name=name,)#  resume=args.resume)
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

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        args.num_classes = 1000

    #####################
    ### Load datasets ###
    #####################
    train_dataset, test_dataset, num_classes, base_c_path, prime_module, aux_dataset = build_dataset(args)
    train_loader, test_loader = build_dataloader(train_dataset, test_dataset, args)
    # if args.aux_type == 'fractals':
    if args.aux_dataset in ['fractals', 'imagenet', 'cifar10']:
        aux_loader = build_auxloader(aux_dataset, args)
    else:
        aux_loader = None

    ####################
    ### Create model ###
    ####################
    net = build_net(args, num_classes)

    #######################
    ### Additional loss ###
    #######################
    additional_loss = AdditionalLoss(args, num_classes, train_loader)

    ###################
    ### Hook layers ###
    ###################
    if args.hook:
        hook = FeatureHook([
                            # "module.relu",
                            "module.avgpool",
                            # "module.proj.3"
                            # "module.block1.layer.5.relu2",
                            # "module.block2.layer.5.relu2",
                            # "module.block3.layer.5.relu2",
                            # 'module.block3',
                            ])
        hook.hook_multi_layer(net)
    elif args.hook_layer != 'None':
        hook = FeatureHook([
                            args.hook_layer
                            ])
        hook.hook_multi_layer(net)

    #################
    ### Optimizer ###
    #################
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)


    ##############
    ### Resume ###
    ##############
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print('Model restored from epoch:', start_epoch)

    #################
    ### Scheduler ###
    #################

    if args.resume and not args.evaluate and not args.analysis:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                (args.epochs - start_epoch) * len(train_loader),
                0.01,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(train_loader),
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))

    #########################
    ### IF: Evaluate mode ###
    #########################
    tester = Tester(net, args, wandb_logger=wandb_logger)
    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc, test_features, test_cm = tester.test(test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc, test_c_table, test_c_cm = tester.test_c(test_dataset, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

        wandb_logger.log_evaluate(dict(test_features=test_features,
                                       test_cm=test_cm,
                                       test_c_table=test_c_table,
                                       test_c_acc=test_c_acc,
                                       test_c_cm=test_c_cm))

        return
    ###########################
    ### ELIF: Analysis mode ###
    ###########################
    elif args.analysis:

        ############################
        ##### class-wise acc #######
        ############################
        test_c_acc, test_c_table, test_c_cm, test_c_features = tester.test_c_cls(test_dataset, base_c_path)  # plot t-sne features
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

        wandb_logger.log_evaluate(dict(test_c_cm=test_c_cm,
                                       test_c_table=test_c_table,
                                       test_c_acc=test_c_acc,
                                       test_c_features=test_c_features
                                       ))

        ##########################
        # test_c_acc, test_c_table, test_c_cms = tester.test_c_save(test_dataset, base_c_path)

        # save false examples of corrupted data
        # test_c_acc, test_c_table, test_c_cms, test_c_cm_mean = tester.test_c_save(test_dataset, base_c_path)

        # train_loss, train_acc, train_features, train_cms = tester.test_v2_trainer(train_loader)
        # wandb_logger.log_evaluate(dict(train_cms=train_cms))
        # #
        # test_loss, test_acc, test_features, test_cm = tester.test(test_loader)
        # test_c_acc, test_c_table, test_c_cms, test_c_features = tester.test_c_v2(test_dataset, base_c_path) # analyzie jsd distance of corrupted data
        # test_c_acc, test_c_table, test_c_cm = tester.test_c(test_dataset, base_c_path)    # plot t-sne features
        # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
        #
        # from utils.visualize import plot_confusion_matrix
        # import matplotlib.pyplot as plt
        # for key, value in test_c_cms.items():
        #     test_c_plt = plot_confusion_matrix(value)
        #     test_c_plt.savefig(f'/ws/data/log/cifar10/debug/{key}.png')

        # wandb_logger.log_evaluate(dict(test_cm=test_cm,
        #                                test_c_table=test_c_table,
        #                                test_c_acc=test_c_acc,
        #                                test_c_cms=test_c_cms,
        #                                # test_c_features=test_c_features
        #                                ))
        return

    ########################
    ### ELSE: Train mode ###
    ########################
    else:
        # ### Scheduler ###
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
        #         step,
        #         args.epochs * len(train_loader),
        #         1,  # lr_lambda computes multiplicative factor
        #         1e-6 / args.learning_rate))

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.isdir(args.save):
            raise Exception('%s is not a dir' % args.save)
        log_path = os.path.join(args.save,
                                args.dataset + '_' + args.model + '_training_log.csv')
        with open(log_path, 'w') as f:
            f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

        ### Trainer ###
        trainer = Trainer(net, args, optimizer, scheduler, wandb_logger=wandb_logger, additional_loss=additional_loss)
        best_acc = 0
        print('Beginning training from epoch:', start_epoch + 1)

        # test_c_acc, test_c_table, test_c_cm = tester.test_c(test_dataset, base_c_path)

        for epoch in range(start_epoch, args.epochs):
            wandb_logger.before_train_epoch() # wandb here
            begin_time = time.time()

            if args.save_grad:
                train_loss_ema, train_features, train_cms = trainer.train_save_grad(train_loader)
            elif args.aug == "prime":
                train_loss_ema, train_features, train_cms = trainer.train_prime(train_loader, prime_module)
            elif args.aux_aug in ['daall']:
                train_loss_ema, train_features, train_cms = trainer.train_auxa(train_loader, aux_loader)
            # elif args.aux_aug in ['da']:
            #     # 0522~
            #     # aug: da enabled
            #     # cifar10 enabled
            #     # aux_label == 'none', 'target', 'uniform'(TODO)
            #     # lw2!=0 ->  logit aux JSD training
            #     # aux_hlambda !=0, hook -> feature aux training
            #     train_loss_ema, train_features, train_cms = trainer.train_auxhd2(train_loader, aux_loader) # 0522~, for da and aux label version
            elif (args.aux_dataset in ['fractals', 'imagenet', 'cifar10']) and (args.aux_hlambda!=0):
                # ~0521
                # aug: unoise (ul and noul), da (no ul) enabled
                # fractals, imagenet, cifar10 enabled
                train_loss_ema, train_features, train_cms = trainer.train_auxhd(train_loader, aux_loader)
            elif args.aux_dataset in ['fractals', 'imagenet']:
                train_loss_ema, train_features, train_cms = trainer.train_auxd(train_loader, aux_loader)
            elif args.aug == 'augda':
                train_loss_ema, train_features, train_cms = trainer.train_augda(train_loader)
            else:
                train_loss_ema, train_features, train_cms = trainer.train(train_loader)

            # wandb_logger.after_train_epoch(dict(train_features=train_features))

            test_loss, test_acc, test_features, test_cm = tester.test(test_loader)
            # wandb_logger.after_test_epoch(d
            # cict(test_features=test_features, # wandb here
            #                                    test_cm=test_cm))

            wandb_logger.log_evaluate(dict(train_features=train_features,
                                           train_cms=train_cms,
                                           test_features=test_features,
                                           test_cm=test_cm,
                                           ))

            ### LOG ###
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

            if not args.save_every:
                save_path = os.path.join(args.save, 'checkpoint.pth.tar')
            else:
                save_path = os.path.join(args.save, f"checkpoint{epoch}.pth.tar")
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

        test_loss, test_acc, _, test_cm = tester.test(test_loader)
        test_c_acc, test_c_table, test_c_cm = tester.test_c(test_dataset, base_c_path)

        wandb_logger.log_evaluate(dict(test_acc=test_acc,
                                       test_cm=test_cm,
                                       test_c_table=test_c_table,
                                       test_c_acc=test_c_acc,
                                       test_c_cm=test_c_cm))

        # wandb_logger.after_run(dict(test_c_table=test_c_table,  # wandb here
        #                             test_c_acc=test_c_acc,
        #                             test_c_cm=test_c_cm))

        print('Clean Error {:.2f}'.format(100 - 100. * test_acc))
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                    (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


        # analysis
        # if args.cls_dg != -1:
        #     test_c_acc, test_c_table, test_c_cm, test_c_features = tester.test_c_cls(test_dataset,
        #                                                                              base_c_path)  # plot t-sne features
        #
        #     wandb_logger.log_evaluate(dict(test_c_cm=test_c_cm,
        #                                    test_c_table=test_c_table,
        #                                    test_c_acc=test_c_acc,
        #                                    test_c_features=test_c_features
        #                                    ))

        # tsne
        # test_tsne = tester.test_c(test_dataset, base_c_path)

if __name__ == '__main__':
    # config_flags.DEFINE_config_file('config') # prime
    main()