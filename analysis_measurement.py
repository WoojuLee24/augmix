from __future__ import print_function

import argparse
import os
import shutil
import time
import torch
import math

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from scipy import linalg
from torchmetrics.functional import structural_similarity_index_measure

from datasets import *
from utils import WandbLogger
from utils.visualize import simple_plot_confusion_matrix

from datasets.builder import simple_build_dataset, simple_build_dataloader
from models import build_net, build_projnet
from models.projnet import projNetv1
from models.losses.projnet_losses import projNetLoss
from datasets.builder import simple_mixed_dataset

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
    # Model
    parser.add_argument('--model', '-m', type=str, default='wrn',
                        choices=['wrn',
                                 'projnetv1',
                                 'wrnproj', 'allconv', 'densenet', 'resnext'],
                        help='Choose architecture.')
    ## projNet architecture options
    parser.add_argument('--hidden-dim', '-hd', default=2048, type=int, help='hidden dims of layers')
    ## WRN Architecture options
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')

    # Optimization options
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    # Checkpointing options
    parser.add_argument('--save', '-s', type=str, default='/ws/data/log', help='Folder to save checkpoints.')
    parser.add_argument('--resume', '-r', type=str, default='', help='Checkpoint path for resume / test.')
    # Acceleration
    parser.add_argument('--num-workers', type=int, default=4, help='Number of pre-fetching threads.')
    # Log
    parser.add_argument('--analysis', action='store_true', default=True, help='Analysis only. ')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    return args


def calculate_fid(act1, act2):
    from third_party.pytorch_fid.fid_score import calculate_frechet_distance
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_score


def calculate_jsd(output1, output2, reduction='sum'):
    assert output1.shape == output2.shape

    output1 = F.softmax(output1, dim=1)
    output2 = F.softmax(output2, dim=1)

    mixture = torch.clamp((output1 + output2) / 2., 1e-7, 1).log()
    jsd = (F.kl_div(mixture, output1, reduction=reduction) +
           F.kl_div(mixture, output2, reduction=reduction)) / 2.

    return jsd


def update_confusion_matrix(outputs, confusion_matrix, categories, y_index):
    x_index = 0
    for category, key_list in categories.items():
        if key_list is None:
            if category == 'fid':
                confusion_matrix[x_index, y_index] = outputs[category] * 1e+19
            elif category == 'ssim':
                confusion_matrix[x_index, y_index] = outputs[category] * 1e+4
            x_index += 1
        else:
            for key in key_list:
                # smaller is better
                if category == 'cossim':
                    confusion_matrix[x_index, y_index] = 1 - abs(outputs[category][key])
                elif category == 'jsd':
                    confusion_matrix[x_index, y_index] = outputs[category][key] * 1e+8
                elif category == 'mse':
                    confusion_matrix[x_index, y_index] = outputs[category][key] * 1e+8
                x_index += 1

    return confusion_matrix


def main():
    args = get_args_from_parser()
    torch.manual_seed(1)
    np.random.seed(1)

    ####################
    ### Create model ###
    ####################
    num_classes = 10
    net = build_projnet(args, num_classes)

    ##############
    ### Resume ###
    ##############
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['state_dict'])
            print('Model restored from epoch:', start_epoch)

    #####################
    ### Load datasets ###
    #####################
    # clean dataset
    clean_dataset = simple_build_dataset(args.dataset, None)
    clean_dataloader = simple_build_dataloader(clean_dataset, args.eval_batch_size, args.num_workers)
    print('Create dataloader for clean dataset')

    # augmented dataset
    aug_kwargs = dict(no_jsd=True, all_ops=True, mixture_width=3, mixture_depth=-1, mixture_coefficient=1.0)
    aug_severities = [0, 1, 2, 3, 4]

    key_list = ['representation', 'projection', 'prediction']
    categories = dict(cossim=key_list, jsd=key_list, mse=key_list, fid=None, ssim=None)

    yticks = ['cossim_re', 'cossim_pj', 'cossim_pd',
              'jsd_re', 'jsd_pj', 'jsd_pd',
              'mse_re', 'mse_pj', 'mse_pd',
              'fid', 'ssim']
    confusion_matrix = torch.zeros(len(yticks), len(aug_severities))
    print(f"confusion matrix ({len(yticks)}, {len(aug_severities)})")

    for aug_severity in aug_severities:
        aug_kwargs.update({'aug_severity': aug_severity})
        augmented_dataset = simple_build_dataset(args.dataset, 'augmix', aug_kwargs=aug_kwargs)

        mixed_aug_dataset = simple_mixed_dataset(clean_dataset, augmented_dataset)
        mixed_aug_dataloader = simple_build_dataloader(mixed_aug_dataset, args.eval_batch_size, args.num_workers)
        print(f'Create dataloader for augmented dataset w/ severity {aug_severity}')

        outputs = analysis(mixed_aug_dataloader, net, args)

        confusion_matrix = update_confusion_matrix(outputs, confusion_matrix,
                                                   categories, y_index=aug_severity)
        print(f"[AugMix] w/ severity {aug_severity}:")
        # print(f"   cossim: {outputs['cossim']['representation']:.8f}, {outputs['cossim']['projection']:.8f}, {outputs['cossim']['prediction']:.8f}")
        # print(f"   jsd: {outputs['jsd']['representation']:.8f}, {outputs['jsd']['projection']:.8f}, {outputs['jsd']['prediction']:.8f}")
        # print(f"   mse: {outputs['mse']['representation']:.8f}, {outputs['mse']['projection']:.8f}, {outputs['mse']['prediction']:.8f}")
        # print(f"   fid: {outputs['fid']:.8f}")
        # print(f"   ssim: {outputs['ssim']:.8f}")
        print(f"   cossim: {outputs['cossim']['representation']:2.4e}, {outputs['cossim']['projection']:2.4e}, {outputs['cossim']['prediction']:2.4e}")
        print(f"   jsd: {outputs['jsd']['representation']:2.4e}, {outputs['jsd']['projection']:2.4e}, {outputs['jsd']['prediction']:2.4e}")
        print(f"   mse: {outputs['mse']['representation']:2.4e}, {outputs['mse']['projection']:2.4e}, {outputs['mse']['prediction']:2.4e}")
        print(f"   fid: {outputs['fid']:2.4e}")
        print(f"   ssim: {outputs['ssim']:2.4e}")

    is_saved = simple_plot_confusion_matrix(confusion_matrix,
                                            normalize=False,
                                            title='The measurement between clean and AugMix w/ severity',
                                            fmt='.3f',
                                            xlabel='', ylabel='',
                                            xticks=['0', '1', '2', '3', '4'],
                                            yticks=yticks,
                                            _range=(0, 1),
                                            save=f'/ws/data/dshong/augmix/measurements/augmix',
                                            )
    if is_saved:
        print(f"Save the measurement result between clean and augmix")
    else:
        print(f"Fail to save the measurement result between clean and augmix")

    # corrupted dataset
    y_index = 0
    confusion_matrix = torch.zeros(len(yticks), len(CORRUPTIONS))
    for corruption in CORRUPTIONS:
        c_kwargs = dict(corruption=corruption,)
        corrupted_dataset = simple_build_dataset(args.dataset, 'corrupted', c_kwargs=c_kwargs)

        mixed_corr_dataset = simple_mixed_dataset(clean_dataset, corrupted_dataset)
        mixed_corr_dataloader = simple_build_dataloader(mixed_corr_dataset, args.eval_batch_size, args.num_workers)
        print(f'Create dataloader for corrupted dataset w/ corruption {corruption}')

        outputs = analysis(mixed_corr_dataloader, net, args)
        confusion_matrix = update_confusion_matrix(outputs, confusion_matrix,
                                                   categories, y_index=y_index)
        y_index += 1

        print(f"[Corruption] w/ {corruption}:")
        # print(f"   cossim: {outputs['cossim']['representation']:.8f}, {outputs['cossim']['projection']:.8f}, {outputs['cossim']['prediction']:.8f}")
        # print(f"   jsd: {outputs['jsd']['representation']:.8f}, {outputs['jsd']['projection']:.8f}, {outputs['jsd']['prediction']:.8f}")
        # print(f"   mse: {outputs['mse']['representation']:.8f}, {outputs['mse']['projection']:.8f}, {outputs['mse']['prediction']:.8f}")
        # print(f"   fid: {outputs['fid']:.8f}")
        # print(f"   ssim: {outputs['ssim']:.8f}")
        print(f"   cossim: {outputs['cossim']['representation']:2.4e}, {outputs['cossim']['projection']:2.4e}, {outputs['cossim']['prediction']:2.4e}")
        print(f"   jsd: {outputs['jsd']['representation']:2.4e}, {outputs['jsd']['projection']:2.4e}, {outputs['jsd']['prediction']:2.4e}")
        print(f"   mse: {outputs['mse']['representation']:2.4e}, {outputs['mse']['projection']:2.4e}, {outputs['mse']['prediction']:2.4e}")
        print(f"   fid: {outputs['fid']:2.4e}")
        print(f"   ssim: {outputs['ssim']:2.4e}")

    is_saved = simple_plot_confusion_matrix(confusion_matrix,
                                            normalize=False,
                                            title='The measurement between clean and corruption',
                                            fmt='.3f',
                                            xlabel='', ylabel='',
                                            xticks=CORRUPTIONS,
                                            yticks=yticks,
                                            _range=(0, 1),
                                            save=f'/ws/data/dshong/augmix/measurements/corruption',
                                            )
    if is_saved:
        print(f"Save the measurement result between clean and corruption")
    else:
        print(f"Fail to save the measurement result between clean and corruption")


def analysis(mixed_loader, model, args):
    """Evaluate network on given dataset."""
    model = model.to(args.device)
    model.eval()

    total_cossim = dict(representation=0.0, projection=0.0, prediction=0.0)
    total_jsd = dict(representation=0.0, projection=0.0, prediction=0.0)
    total_mse = dict(representation=0.0, projection=0.0, prediction=0.0)
    total_fid = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for i, (x_clean, y_clean, x_target, y_target) in enumerate(mixed_loader):
            x_clean, y_clean = x_clean.to(args.device), y_clean.to(args.device)
            x_target, y_target = x_target.to(args.device), y_target.to(args.device)

            # outputs: representation=(B,128,1,1), projection=(B,hidden), prediction=(B,num_classes)
            outputs_clean = model(x_clean)
            outputs_clean['representation'] = outputs_clean['representation'].squeeze()

            outputs_target = model(x_target)
            outputs_target['representation'] = outputs_target['representation'].squeeze()

            # Cosine similarity: (B,)
            # : 1 if totally same, 0 if 90', -1 if totally opposite directed. Bounds=[-1,1]
            key_list = ['representation', 'projection', 'prediction']
            for key in key_list:
                cossim = F.cosine_similarity(outputs_clean[key], outputs_target[key], dim=1)
                cossim = torch.sum(cossim)
                total_cossim[key] += float(cossim)

            # JSD(Jensen-Shannon Divergence)
            # : Smaller is similar . Bounds=[0,1]
            key_list = ['representation', 'projection', 'prediction']
            for key in key_list:
                jsd = calculate_jsd(outputs_clean[key], outputs_target[key], reduction='none')
                jsd = torch.sum(torch.sum(jsd, dim=1))
                total_jsd[key] += float(jsd)

            # MSE(Mean-Square Error)
            # : Smaller is similar.
            key_list = ['representation', 'projection', 'prediction']
            for key in key_list:
                mse = F.mse_loss(outputs_clean[key], outputs_target[key], reduction='none')
                mse = torch.sum(torch.sum(mse, dim=1))
                total_mse[key] += float(mse)

            # FID (Frechet Inception Divergence)
            # : Smaller is similar. Bounds=[0,infinite)
            fid = calculate_fid(outputs_clean['representation'].cpu().numpy(),
                                outputs_target['representation'].cpu().numpy())

            total_fid += float(fid)

            # SSIM
            # :
            ssim = structural_similarity_index_measure(x_clean, x_target)
            total_ssim += float(ssim)

        size = len(mixed_loader.dataset)
        key_list = ['representation', 'projection', 'prediction']
        final_cossim, final_jsd, final_mse = dict(), dict(), dict()
        for key in key_list:
            final_cossim[key] = total_cossim[key] / size
            final_jsd[key] = total_jsd[key] / size
            final_mse[key] = total_mse[key] / size
        final_fid = total_fid / size
        final_ssim = total_ssim / size

        outputs = dict(cossim=final_cossim,
                       jsd=final_jsd,
                       mse=final_mse,
                       fid=final_fid,
                       ssim=final_ssim)

    return outputs


if __name__ == '__main__':
    main()