from __future__ import print_function

import argparse
import os
import shutil
import time
import torch
import math

import numpy as np
import torch.nn.functional as F

from datasets import *
from utils import WandbLogger
from utils.visualize import plot_confusion_matrix

from models.projnet import projNetv1
from models.losses.projnet_losses import projNetLossv1

def get_args_from_parser():
    parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--aug', '-aug',
                        type=str,
                        default='augmix',
                        choices=['none', 'augmix', 'pixmix', 'apr'],
                        help='Choose domain generalization augmentation methods')
    ## AugMix options
    parser.add_argument('--mixture-width', default=3, type=int,
                        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument('--mixture-depth', default=-1, type=int,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
    parser.add_argument('--mixture-coefficient', '-mc', default=1.0, type=float, help='mixture coefficient alpha')
    parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
    parser.add_argument('--additional-loss', '-al',
                        default='jsd',
                        type=str,
                        choices=['none', 'jsd', 'jsd_temper', 'jsdv3', 'jsdv3.01', 'jsdv3.02', 'kl',
                                 'supconv0.01',
                                 'ntxent', 'center_loss', 'mlpjsd', 'mlpjsdv1.1'],
                        help='Type of additional loss')
    parser.add_argument('--temper', default=1.0, type=float, help='temperature scaling')
    parser.add_argument('--lambda-weight', '-lw', default=12.0, type=float, help='additional loss weight')
    parser.add_argument('--reduction', default='batchmean', type=str, choices=['batchmean', 'mean'],
                        help='temperature scaling')
    parser.add_argument('--margin', default=0.02, type=float, help='triplet loss margin')
    parser.add_argument('--jsd-layer', default='features', type=str, choices=['features', 'logits'],
                        help='apply jsd loss for the selected layer')
    parser.add_argument('--hook', action='store_true', help='hook layers for feature extraction')
    parser.add_argument('--all-ops', '-all', action='store_true',
                        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    ## PIXMIX options
    parser.add_argument('--beta', default=3, type=int, help='Severity of mixing')
    parser.add_argument('--k', default=4, type=int, help='Mixing iterations')
    parser.add_argument('--mixing-set', type=str, default='/ws/data/fractals_and_fvis/', help='Mixing set directory.')
    parser.add_argument('--use_300k', action='store_true', help='use 300K random images as aug data')

    # Model
    parser.add_argument('--model', '-m',
                        type=str,
                        default='wrn',
                        choices=['wrn', 'wrnproj', 'allconv', 'densenet', 'resnext'],
                        help='Choose architecture.')
    ## WRN Architecture options
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

    # Checkpointing options
    parser.add_argument('--save', '-s', type=str, default='/ws/data/log', help='Folder to save checkpoints.')
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

    args = parser.parse_args()

    return args

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:min(k, maxk)].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = get_args_from_parser()
    torch.manual_seed(1)
    np.random.seed(1)

    ########################
    ### Initialize wandb ###
    ########################
    name = 'projNetv1'

    if args.wandb:
        wandg_config = dict(project='Classification', entity='kaist-url-ai28', name=name)
        wandb_logger = WandbLogger(wandg_config, args)
    else:
        wandb_logger = WandbLogger(None)
    wandb_logger.before_run()

    #####################
    ### Load datasets ###
    #####################
    train_dataset, test_dataset, num_classes, base_c_path = build_dataset(args)
    train_loader, test_loader = build_dataloader(train_dataset, test_dataset, args)

    ####################
    ### Create model ###
    ####################
    net = projNetv1(pred_dim=num_classes, args=args, hidden_dim=128)
    criterion = projNetLossv1(lambda_weight=1e-1)

    #################
    ### Optimizer ###
    #################
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    ### Scheduler ###
    def get_lr(step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                                   np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

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

    #############
    ### Train ###
    #############
    for epoch in range(start_epoch, args.epochs):

        train(train_loader, net, criterion, optimizer, scheduler, epoch, args, wandb_logger)

        ### LOG ###
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.model,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(args.save, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, wandb_logger):
    device = 'cuda'
    model = model.to(device)
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        data_time = time.time() - end

        optimizer.zero_grad()

        images_all = torch.cat(images, 0).to(device)
        targets = targets.to(device)

        # 'outputs' have representation, projection, and prediction
        outputs_all = model(images_all)

        outputs_clean, outputs_aug1, outputs_aug2 = dict(), dict(), dict()
        for key, value in outputs_all.items():
             clean, aug1, aug2 = torch.split(value, images[0].size(0))
             outputs_clean.update({key: clean})
             outputs_aug1.update({key: aug1})
             outputs_aug2.update({key: aug2})
        del outputs_all

        # TODO: Please design your loss function
        loss = criterion(outputs_clean, outputs_aug1, outputs_aug2, targets=targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(outputs_clean['prediction'], targets, topk=(1, 5))

        type = 'train'
        wandb_logger.log(f'{type}/loss', float(loss))
        wandb_logger.log(f'{type}/ce_loss', float(criterion.outputs['original_loss']))
        wandb_logger.log(f'{type}/additional_loss',
                         float(criterion.lambda_weight * criterion.outputs['additional_loss']))
        wandb_logger.log(f'{type}/acc1', float(acc1))

        end = time.time()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()