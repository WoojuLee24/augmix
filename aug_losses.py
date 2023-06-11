import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_aug_loss(args, logits, targets, lambda_weight=12):

    name = args.aug_loss
    if name == 'none':
        loss = 0
    elif name == 'msel1':
        loss, features = msel1(args, logits, targets)
    elif name == 'msel1ssim':
        loss, features = msel1ssim(args, logits, targets)


    return loss, features


def msel1(args, logits, targets):
    # default: alw=0.1, alw2=0.5
    features = dict()
    mse_loss = F.mse_loss(logits, targets)
    l1_loss = F.l1_loss(logits, targets)
    ssim = get_ssim(args, logits, targets)

    loss = args.alw * mse_loss + args.alw2 * l1_loss
    features['loss'] = loss.detach()
    features['mse_loss'] = mse_loss.detach()
    features['l1_loss'] = l1_loss.detach()
    features['ssim'] = ssim.detach()

    return loss, features

def msel1ssim(args, logits, targets):
    # default: alw=0.1, alw2=0.5, alw3=?
    features = dict()
    mse_loss = F.mse_loss(logits, targets)
    l1_loss = F.l1_loss(logits, targets)
    ssim = get_ssim(args, logits, targets)

    loss = args.alw * mse_loss + args.alw2 * l1_loss + args.alw3 * -1 * ssim
    features['loss'] = loss.detach()
    features['mse_loss'] = mse_loss.detach()
    features['l1_loss'] = l1_loss.detach()
    features['ssim'] = ssim.detach()

    return loss, features



def csl2(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper=1, reduction='mean'):

    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),

    sim1 = F.cosine_similarity(logits_clean, logits_aug1).pow(2)
    sim2 = F.cosine_similarity(logits_aug1, logits_aug2).pow(2)
    sim3 = F.cosine_similarity(logits_aug2, logits_clean).pow(2)
    logits = 1 - (sim1 + sim2 + sim3) / 3
    loss = logits.mean() / temper * lambda_weight
    features = {'distance': logits.mean().detach()}

    return loss, features


def cslp(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper=2, reduction='mean'):

    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),

    sim1 = F.cosine_similarity(logits_clean, logits_aug1).pow(temper)
    sim2 = F.cosine_similarity(logits_aug1, logits_aug2).pow(temper)
    sim3 = F.cosine_similarity(logits_aug2, logits_clean).pow(temper)
    logits = 1 - (sim1 + sim2 + sim3) / 3
    loss = logits.mean() / temper * lambda_weight
    features = {'distance': logits.mean().detach()}

    return loss, features


def msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    # collapse with lw 12
    B, C = logits_clean.size()

    distance = (F.mse_loss(logits_clean, logits_aug1, reduction='sum') +
                F.mse_loss(logits_clean, logits_aug2, reduction='sum') +
                F.mse_loss(logits_aug1, logits_aug2, reduction='sum')) / 3 / B


    loss = lambda_weight * distance
    feature = {'distance': distance.detach()}

    return loss, feature


def cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper=1, reduction='mean'):

    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),

    sim1 = F.cosine_similarity(logits_clean, logits_aug1)
    sim2 = F.cosine_similarity(logits_aug1, logits_aug2)
    sim3 = F.cosine_similarity(logits_aug2, logits_clean)
    logits = 1 - (sim1 + sim2 + sim3) / 3
    loss = logits.mean() / temper * lambda_weight
    features = {'distance': logits.mean().detach()}

    return loss, features



from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, reduction):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    B, C, H, W = ssim_map.size()
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'batchmean':
        return ssim_map.sum() / B / H / W
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(args, img_clean, img_aug1, img_aug2, lambda_weight, targets, temper=1, reduction='mean'):

    window_size = args.window
    (_, channel, _, _) = img_clean.size()
    window = create_window(window_size, channel)
    if img_clean.is_cuda:
        window = window.cuda(img_clean.get_device())
    window = window.type_as(img_clean)

    loss1 = _ssim(img_clean, img_aug1, window, window_size, channel, reduction)
    loss2 = _ssim(img_aug1, img_aug2, window, window_size, channel, reduction)
    loss3 = _ssim(img_aug2, img_clean, window, window_size, channel, reduction)

    loss = 1 - (loss1 + loss2 + loss3) / 3
    features = {'distance': loss.detach()}
    loss = loss * lambda_weight

    return loss, features


def get_ssim(args, img_clean, img_aug1, reduction='mean'):
    window_size = args.window
    sigma = args.sigma # sigma for gaussian window
    (_, channel, _, _) = img_clean.size()
    window = create_window(window_size, channel, sigma)
    if img_clean.is_cuda:
        window = window.cuda(img_clean.get_device())
    window = window.type_as(img_clean)

    ssim = _ssim(img_clean, img_aug1, window, window_size, channel, reduction)
    return ssim

