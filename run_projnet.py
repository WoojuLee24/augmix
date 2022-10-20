from __future__ import print_function

import argparse
import os
import shutil
import time
import torch
import math

import numpy as np
import pandas as pd
import torch.nn.functional as F

from datasets import *
from utils import WandbLogger
from utils.visualize import plot_confusion_matrix

from models import build_net, build_projnet
from models.projnet import projNetv1
from models.losses.projnet_losses import projNetLoss

CORRUPTIONS = [
  'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
  'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
  'brightness', 'contrast', 'elastic_transform', 'pixelate',
  'jpeg_compression'
]

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
                                 'projnetv1.pred', 'projnetv1.proj', 'projnetv1.repr',
                                 'projnetv1.1', 'projnetv1.2', 'projnetv1.3',
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
                        choices=['wrn',
                                 'projnetv1', 'projnetv1.1', 'projnetv1.2', 'projnetv1.3', 'projnetv1.4', 'projnetv1.5',
                                 'projnetv1.6', 'projnetv1.1.1', 'projnetv1.2.1', 'projnetv1.3.1', 'projnetv1.4.1', 'projnetv1.5.1',
                                 'projnetv1.1.2', 'projnetv1.1.3', 'projnetv1.6.1', 'projnetv1.6.2',
                                 'wrnproj', 'allconv', 'densenet', 'resnext'],
                        help='Choose architecture.')
    ## projNet architecture options
    parser.add_argument('--hidden-dim', '-hd', default=2048, type=int, help='hidden dims of layers')

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
    # name = 'projNetv1'

    save_path = args.save
    name = save_path.split("/")[-1]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

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
    # net = projNetv1(pred_dim=num_classes, args=args, hidden_dim=128)
    net = build_projnet(args, num_classes)
    criterion = projNetLoss(name=args.additional_loss, lambda_weight=args.lambda_weight)
    criterion_test = projNetLoss(name='projnetv1.test', lambda_weight=args.lambda_weight)

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
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)

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

    #########################
    ### IF: Evaluate mode ###
    #########################

    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_error, test_features, test_cm = test(test_loader, net, criterion_test, args, wandb_logger)
        print(f'Clean: Test Loss: {float(test_loss)} | Test error: {test_error}')

        # Evaluate corruption accuracy
        test_c_error, wandb_features, test_c_table, test_c_cm = test_c(test_dataset, net, criterion_test, args, wandb_logger,
                                                                      base_path='/ws/data/CIFAR/CIFAR-10-C/')
        print(f'Mean Corruption Error: {test_c_error}')

        wandb_logger.log_evaluate(dict(test_features=test_features,
                                       test_cm=test_cm,
                                       test_c_table=test_c_table,
                                       test_c_error=test_c_error,
                                       test_c_cm=test_c_cm))

        return

    ###########################
    ### ELIF: Analysis mode ###
    ###########################

    if args.analysis:

        return

    #############
    ### Train ###
    #############

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):

        train_loss, train_acc, train_features = train(train_loader, net, criterion, optimizer, scheduler, epoch, args, wandb_logger)
        test_loss, test_acc, test_features, test_cm = test(test_loader, net, criterion_test, args, wandb_logger)

        wandb_logger.log_evaluate(dict(train_features=train_features,
                                       test_features=test_features,
                                       test_cm=test_cm))

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

        for param_group in optimizer.param_groups:
            wandb_logger.log('learng_rate', param_group['lr'])

        save_path = os.path.join(args.save, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    # Evaluate corruption accuracy debug
    test_c_error, test_c_features, test_c_table, test_c_cm = test_c(test_dataset, net, criterion_test, args,
                                                                    wandb_logger,
                                                                    base_path='/ws/data/CIFAR/CIFAR-10-C/')

    wandb_logger.log_evaluate(dict(test_c_error=test_c_error,
                                   # test_c_features=test_c_features,
                                   test_c_table=test_c_table,
                                   test_c_cm=test_c_cm))


def train(data_loader, model, criterion, optimizer, scheduler, epoch, args, wandb_logger):
    device = 'cuda'
    model = model.to(device)
    model.train()

    # logging
    type = 'train'
    total_loss, total_correct = 0., 0.
    wandb_features = dict()

    end = time.time()
    for i, (images, targets) in enumerate(data_loader):
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

        # log criterion features: ce_loss, additional_loss, features,..
        if i == 0:
            for key, value in criterion.outputs.items():
                wandb_features[f'{type}/{key}'] = value.detach()
        else:
            if outputs_clean['prediction'].size(0) == args.batch_size:
                for key, value in criterion.outputs.items():
                    wandb_features[f'{type}/{key}'] += value.detach()

        # acc1, acc5 = accuracy(outputs_clean['prediction'], targets, topk=(1, 5))
        pred = outputs_clean['prediction'].data.max(1)[1]
        acc = pred.eq(targets.data).sum().item() / args.batch_size
        error = 100 - 100 * acc

        total_loss += float(loss.data)
        total_correct += pred.eq(targets.data).sum().item()
        end = time.time()

        # if i % args.print_freq == 0:
        #     print(f'Batch {i}/{len(data_loader)}: Train Loss: {float(loss.data)} | Train error: {error}')

    # logging total results
    # features
    denom = math.floor(len(data_loader.dataset) / args.batch_size)
    for key, value in wandb_features.items():
        wandb_features[key] = value / denom
    # loss, error
    denom = len(data_loader.dataset) / args.batch_size
    train_loss = total_loss / denom
    datasize = len(data_loader.dataset)
    train_error = 100 - 100. * total_correct / datasize
    wandb_features[f'{type}/total_loss_epoch'] = train_loss
    wandb_features[f'{type}/error'] = train_error

    print(f'Epoch {epoch}: Train Loss: {train_loss: .3f} | Train error: {train_error: .3f}')

    # if args.wandb:
    #     for key, value in wandb_features.items():
    #         wandb_logger.log(key, value)

    return train_loss, train_error, wandb_features

def test(data_loader, model, criterion, args, wandb_logger, data_type='test/'):
    """Evaluate network on given dataset."""

    device = 'cuda'
    model = model.to(device)
    model.eval()

    total_loss, total_correct = 0., 0.
    wandb_features = dict()
    confusion_matrix = torch.zeros(10, 10)
    tsne_features = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.cuda(), targets.cuda()
            outputs_all = model(images)
            # outputs_clean, outputs_aug1, outputs_aug2 = dict(), dict(), dict()
            # for key, value in outputs_all.items():
            #     clean, aug1, aug2 = torch.split(value, images[0].size(0))
            #     outputs_clean.update({key: clean})
            #     outputs_aug1.update({key: aug1})
            #     outputs_aug2.update({key: aug2})
            # del outputs_all

            # TODO: Please design your loss function
            loss = criterion(outputs_all, targets=targets)

            # log criterion features: ce_loss, additional_loss, features,..
            if i == 0:
                for key, value in criterion.outputs.items():
                    wandb_features[f'{data_type}{key}'] = value.detach()
            else:
                if outputs_all['prediction'].size(0) == args.eval_batch_size:
                    for key, value in criterion.outputs.items():
                        wandb_features[f'{data_type}{key}'] += value.detach()

            # acc1, acc5 = accuracy(outputs_all['prediction'], targets, topk=(1, 5))
            pred = outputs_all['prediction'].data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            if args.analysis:
                from utils.visualize import multi_plot_tsne
                input_list = [model.module.features, logits]
                targets_list = [targets, targets]
                title_list = ['features', 'logits']
                save_path = os.path.join(args.save, 'analysis', data_type + '.jpg')
                tsne, fig = multi_plot_tsne(input_list, targets_list, title_list, rows=1, cols=2,
                                            perplexity=30, n_iter=300,
                                            save=save_path, log_wandb=args.wandb, data_type=data_type)

    # logging total results
    # features
    denom = math.floor(len(data_loader.dataset) / args.eval_batch_size)
    for key, value in wandb_features.items():
        wandb_features[key] = value / denom

    # loss
    denom = len(data_loader.dataset) / args.eval_batch_size
    test_loss = total_loss / denom
    # wandb_features[f'{data_type}loss_epoch'] = test_loss

    # error
    datasize = len(data_loader.dataset)
    test_error = 100 - 100. * total_correct / datasize
    wandb_features[f'{data_type}error'] = test_error

    print(f'{data_type} Test Loss: {test_loss: .3f} | Test Error: {test_error: .3f}')

    # if args.wandb:
    #     for key, value in wandb_features.items():
    #         wandb_logger.log(key, value)

    return test_loss, test_error, wandb_features, confusion_matrix


def test_c(test_dataset, model, criterion, args, wandb_logger, base_path=None):
    """Evaluate network on given corrupted dataset."""
    wandb_features, wandb_plts = dict(), dict()
    wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
    confusion_matrices = []
    if args.dataset == 'cifar10' or 'cifar100':
        corruption_errors = []
        for corruption in CORRUPTIONS:
            # Reference to original data is mutated
            test_dataset.data = np.load(base_path + corruption + '.npy')
            test_dataset.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True)
            data_type = f'test/{corruption}/'
            test_loss, test_error, test_feature, confusion_matrix = test(test_loader, model, criterion, args, wandb_logger, data_type)

            wandb_features.update(test_feature)
            wandb_table[corruption]['loss'] = test_loss
            wandb_table[corruption]['error'] = test_error
            # wandb_plts[corruption] = confusion_matrix

            corruption_errors.append(test_error)
            confusion_matrices.append(confusion_matrix.cpu().detach().numpy())
            # print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            #     corruption, test_loss, test_error))

        # return np.mean(corruption_accs), wandb_features
        test_c_error = np.mean(corruption_errors)
        test_c_cm = np.mean(confusion_matrices, axis=0)
        print(f'Mean Corruption Error: {test_c_error: .3f}')

        return test_c_error, wandb_features, wandb_table, test_c_cm


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()