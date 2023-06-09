import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_additional_loss(args, logits_clean, logits_aug1, logits_aug2,
                        lambda_weight=12, targets=None, temper=1, reduction='batchmean', **kwargs):

    name = args.additional_loss
    if name == 'none':
        loss = 0
    elif name == 'jsd':
        loss, features = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.3':
        loss, features = kl_v1_3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, args.skew)
    elif name == 'msev1.0':
        loss, features = msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'analysisv1.0':  # analysis test mode
        loss, features = analysisv1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight)

    return loss, features


def get_additional_loss2(args, logits_clean, logits_aug1, logits_aug2,
                        lambda_weight=12, targets=None, temper=1, reduction='batchmean', **kwargs):

    name = args.additional_loss
    if name == 'none':
        loss = 0
    elif name == 'jsd':
        loss, features = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.3':
        loss, features = kl_v1_3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, args.skew)
    elif name == 'msev1.0':
        loss, features = msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'cslp':
        loss, features = cslp(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper=args.temper, reduction='mean')
    elif name == 'analysisv1.0':  # analysis test mode
        loss, features = analysisv1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight)

    return loss, features


def analysisv1_0(logits_clean, logits_aug1, logits_aug2=None, lambda_weight=12):

    B, C = logits_clean.size()

    p_clean, p_aug1, = F.softmax(logits_clean, dim=1),\
                       F.softmax(logits_aug1, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    # jsd: batchmean reduction
    p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()
    jsd = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1, reduction='batchmean')) / 2.
    jsd_mean = (F.kl_div(p_mixture, p_clean, reduction='mean') + F.kl_div(p_mixture, p_aug1, reduction='mean')) / 2.

    # mse: mean reduction
    mse = (F.mse_loss(logits_clean, logits_aug1, reduction='mean')) / B

    # cosine_similarity: mean reduction
    similarity = F.cosine_similarity(logits_clean, logits_aug1)
    similarity = similarity.mean()

    features = {'jsd_batchmean': jsd,
                'jsd_mean': jsd_mean,
                'mse': mse,
                'similarity': similarity,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_mixture': p_mixture,
                }
    loss = jsd

    return loss, features


def jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance.detach(),
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                # 'p_aug2': p_aug2,
                }

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


def kl_v1_3(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, skew=0.8):
    """
    from klv1.2
    skew kl divergence
    """

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_aug1_skew = (1 - skew) * p_aug1 + skew * p_clean
    p_aug2_skew = (1 - skew) * p_aug2 + skew * p_clean

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_skew_log = torch.clamp(p_aug1_skew, 1e-7, 1).log()
    p_aug2_skew_log = torch.clamp(p_aug2_skew, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_skew_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_skew_log, p_clean, reduction='batchmean')) / 2.

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    features = {'jsd_distance': jsd_distance.detach(),
                }

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


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
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

def ssim_multi(args, imgs, auxn, lambda_weight=12, temper=1, reduction='mean'):

    window_size = args.window
    auxtotal = imgs.size(0)
    auxb = auxtotal // auxn

    (_, channel, height, width) = imgs.size()
    imgs = imgs.reshape(auxn, auxb, channel, height, width)
    # (_, channel, _, _) = img_clean.size()
    window = create_window(window_size, channel)
    if imgs.is_cuda:
        window = window.cuda(imgs.get_device())
    window = window.type_as(imgs)

    ssims = 0

    for i in range(1, auxb):
        a = imgs[:, 0]
        b = imgs[:, i]
        ssims += _ssim(imgs[:, 0], imgs[:, i], window, window_size, channel, reduction)

    ssims = ssims / auxb

    loss = 1 - ssims

    features = {'distance': loss.detach()}
    loss = loss * lambda_weight

    return loss, features