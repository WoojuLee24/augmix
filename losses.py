import torch
import torch.nn.functional as F
import torch.nn as nn



def get_additional_loss(args, logits_clean, logits_aug1, logits_aug2,
                        lambda_weight=12, targets=None, temper=1, reduction='batchmean', **kwargs):

    name = args.additional_loss
    if name == 'none':
        loss = 0
    elif name == 'jsd':
        loss, features = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
        return loss, features
    elif name == 'jsdvl_v0.1':
        loss, features = jsdvl_v0_1(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper)
        return loss, features
    elif name == 'jsdvl_v0.1.1':
        alpha = args.lambda_alpha
        beta = args.lambda_beta
        loss, features = jsdvl_v0_1_1(logits_clean, logits_aug1, logits_aug2, targets, alpha, beta, lambda_weight, temper)
        return loss, features
    elif name == 'jsd.skew':
        loss, features = jsd_skew(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, args.skew)
        return loss, features
    elif name == 'jsd.manual':
        loss = jsd_manual(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsd.manual.ce':
        loss = jsd_manual_ce(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsdv1':
        loss = jsdv1(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsdv2':
        loss = jsdv2(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'jsdv2.1':
        loss = jsdv2_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'jsdv3':
        loss, features = jsdv3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv4':
        loss, features = jsdv4(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.test':
        loss, features = jsdv3_test(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.cossim':
        loss, features = jsdv3_cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.msev1.0':
        margin = args.margin
        loss, features = jsdv3_msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.msev1.0.detach':
        margin = args.margin
        loss, features = jsdv3_msev1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.msev1.1':
        margin = args.margin
        loss, features = jsdv3_msev1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.simsiam':
        loss, features = jsdv3_simsiam(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.simsiamv0.1':
        loss, features = jsdv3_simsiamv0_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.0.1':
        margin = args.margin
        loss, features = jsdv3_0_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.0.2':
        margin = args.margin
        loss, features = jsdv3_0_2(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.0.3':
        margin = args.margin
        loss, features = jsdv3_0_3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.0.1.detach':
        margin = args.margin
        loss, features = jsdv3_0_1_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.0.2.detach':
        margin = args.margin
        loss, features = jsdv3_0_2_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.0.3.detach':
        margin = args.margin
        loss, features = jsdv3_0_3_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.log.inv':
        loss, features = jsdv3_log_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.inv':
        loss, features = jsdv3_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsd.ntxent':
        margin = args.margin
        loss, features = jsd_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsd.ntxentv0.01':
        margin = args.margin
        loss, features = jsd_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsd.ntxentv0.02':
        margin = args.margin
        loss, features = jsd_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.ntxent':
        margin = args.margin
        loss, features = jsdv3_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.ntxentv0.01':
        margin = args.margin
        loss, features = jsdv3_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.ntxentv0.02':
        margin = args.margin
        loss, features = jsdv3_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets,
                                           margin)
        return loss, features
    elif name == 'jsdv3.ntxent.detach':
        margin = args.margin
        loss, features = jsdv3_ntxent_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsd_temper' or name == 'mlpjsdv1.1':
        loss = jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, reduction)
    elif name == 'jsd_kd':
        loss = jsd_kd(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, reduction)
    elif name == 'jsd_kdv2':
        loss = jsd_kdv2(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, reduction)
    elif name == 'kl':
        loss = kl(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'cossim':
        loss = cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper, reduction)
    elif name == 'ntxent':
        loss = ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets)
    elif name == 'supcontrast':
        loss = supcontrast(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction)
    elif name == 'supcontrast_test':
        loss = supcontrast_test(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction)
    elif name == 'supcontrastv0.01':
        loss = supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight,  temper, reduction)
    elif name == 'supcontrastv0.02':
        loss = supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight,  temper, reduction)
    elif name == 'jsdv3_apr_p':
        loss, features = jsdv3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features

    # kl_div
    elif name == 'klv1.0':
        loss, features = kl_v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
        return loss, features
    elif name == 'klv1.1':
        loss, features = kl_v1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
        return loss, features
    elif name == 'klv1.2':
        loss, features = kl_v1_2(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
        return loss, features
    elif name == 'klv1.3':
        loss, features = kl_v1_3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, args.skew)
        return loss, features
    elif name == 'klv1.0.detach':
        loss = kl_v1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.1.detach':
        loss = kl_v1_1_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.2.detach':
        loss = kl_v1_2_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.1.inv':
        loss = kl_v1_1_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.2.inv':
        loss = kl_v1_2_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)

    # mse
    elif name == 'msev1.0':
        loss = msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'msev1.0.detach':
        loss = mse_v1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)

    # analysis test mode
    elif name == 'analysisv1.0':
        loss, features = analysisv1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight)
        return loss, features

    return loss


def get_additional_loss2(args, logits_clean, logits_aug1, logits_aug2,
                        lambda_weight=12, targets=None, temper=1, reduction='batchmean', **kwargs):

    name = args.additional_loss2
    if name == 'none':
        loss, features = 0, dict()
    elif name == 'cossim':
        loss, features = cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper, reduction)
    elif name == 'ssim':
        loss, features = ssim(args, logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper, reduction)
    elif name == 'msev1.1':
        loss, features = msev1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'msev1.0':
        loss, features = msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsd':
        loss, features = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'jsdv4.ntxent':
        margin = args.margin
        loss, features = jsdv4_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
    elif name == 'jsdv4.ntxentv0.01':
        margin = args.margin
        loss, features = jsdv4_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
    elif name == 'jsdv4.ntxentv0.02':
        margin = args.margin
        loss, features = jsdv4_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
    elif name == 'jsdv4.ntxent.detach':
        margin = args.margin
        loss, features = jsdv4_ntxent_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
    elif name == 'opl':
        loss, features = opl(logits_clean, logits_aug1, logits_aug2, args, lambda_weight, temper, labels=targets)

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


def jsdvl_v0_1(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight=12, temper=1.0):
    """
    variational weight w.r.t correct class probability
    """

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    vl = p_clean[torch.arange(p_clean.size(0)), targets]
    vl = vl.detach()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    batch_size = p_clean.size(0)
    jsd_distance_ = (F.kl_div(p_mixture, p_clean, reduction='none').sum(1) * vl +
                    F.kl_div(p_mixture, p_aug1, reduction='none').sum(1) * vl +
                    F.kl_div(p_mixture, p_aug2, reduction='none').sum(1) * vl)
    jsd_distance = jsd_distance_.sum(0) / 3. / batch_size

    # jsd_distance2_ = (F.kl_div(p_mixture, p_clean, reduction='none').sum(1) +
    #                  F.kl_div(p_mixture, p_aug1, reduction='none').sum(1) +
    #                  F.kl_div(p_mixture, p_aug2, reduction='none').sum(1))
    # jsd_distance2 = jsd_distance2_.sum(0) / 3. / batch_size
    #
    # jsd_distance3 = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance,
                }

    return loss, features

def jsdvl_v0_1_1(logits_clean, logits_aug1, logits_aug2, targets, alpha=0.5, beta=1, lambda_weight=12, temper=1.0):
    """
    variational weight w.r.t correct class probability
    correct class prob are shifted by alpha. m <= p <= 1 + m, and length is controlled by beta
    """

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    vl = p_clean[torch.arange(p_clean.size(0)), targets]
    vl = vl * (beta - alpha) + alpha
    vl = vl.detach()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    batch_size = p_clean.size(0)
    jsd_distance_ = (F.kl_div(p_mixture, p_clean, reduction='none').sum(1) * vl +
                    F.kl_div(p_mixture, p_aug1, reduction='none').sum(1) * vl +
                    F.kl_div(p_mixture, p_aug2, reduction='none').sum(1) * vl)
    jsd_distance = jsd_distance_.sum(0) / 3. / batch_size

    # jsd_distance2_ = (F.kl_div(p_mixture, p_clean, reduction='none').sum(1) +
    #                  F.kl_div(p_mixture, p_aug1, reduction='none').sum(1) +
    #                  F.kl_div(p_mixture, p_aug2, reduction='none').sum(1))
    # jsd_distance2 = jsd_distance2_.sum(0) / 3. / batch_size
    #
    # jsd_distance3 = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance,
                }

    return loss, features


def jsd_skew(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, skew=0.8):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                            F.softmax(logits_aug2 / temper, dim=1)

    # p_clean_skew = skew * p_clean
    p_aug1_skew = (1 - skew) * p_aug1 + skew * p_clean
    p_aug2_skew = (1 - skew) * p_aug2 + skew * p_clean

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1_skew + p_aug2_skew) / 3., 1e-7, 1).log()
    jsd_distance_skew = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1_skew, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2_skew, reduction='batchmean')) / 3.

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss = lambda_weight * jsd_distance_skew

    features = {'jsd_distance': jsd_distance,
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                # 'p_aug2': p_aug2,
                }

    return loss, features


def jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsd_manual(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    B, C = logits_clean.size()
    loss = ((p_clean * p_clean.log() - p_clean * p_mixture) + (p_aug1 * p_aug1.log() - p_aug1 * p_mixture) + (p_aug2 * p_aug2.log() - p_aug2 * p_mixture))
    loss = lambda_weight * loss.sum() / 3 / B

    return loss

def jsd_manual_ce(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    B, C = logits_clean.size()
    loss = ((- p_clean * p_mixture) + (- p_aug1 * p_mixture) + (- p_aug2 * p_mixture))
    loss = lambda_weight * loss.sum() / 3 / B / C

    return loss

def jsdv1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # No clamp mixture distribution
    p_mixture = ((p_clean + p_aug1 + p_aug2) / 3).log()
    loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return loss

def jsdv2(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1):
    '''
    JSD loss edited: mixture probability is prediction probability
    '''
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3
    loss = lambda_weight * (F.kl_div(p_clean.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug1.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug2.log(), p_mixture, reduction='batchmean')) / 3.

    return loss


def jsdv2_1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1):
    '''
    JSD loss edited:
    optimize max(JSD(p, m), JSD(m, p))
    '''
    B, C = logits_clean.size()
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3

    loss = (F.kl_div(p_clean.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug1.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug2.log(), p_mixture, reduction='batchmean')) / 3.


    jsd_distance = (F.kl_div(p_clean.log(), p_mixture, reduction='none') +
                    F.kl_div(p_aug1.log(), p_mixture, reduction='none') +
                    F.kl_div(p_aug2.log(), p_mixture, reduction='none')) / 3.

    jsd_distance_inv = (F.kl_div(p_mixture.log(), p_clean, reduction='none') +
                        F.kl_div(p_mixture.log(), p_aug1, reduction='none') +
                        F.kl_div(p_mixture.log(), p_aug2, reduction='none')) / 3.

    jsd_distance = jsd_distance.sum(-1)
    jsd_distance_inv = jsd_distance_inv.sum(-1)

    jsd_distance_max = torch.maximum(jsd_distance, jsd_distance_inv)

    jsd_distance_max = jsd_distance_max.sum() / B

    loss = lambda_weight * jsd_distance_max

    return loss


def get_kl_matrix(p, q):
    B, C = p.size()
    ent = p * p.log()
    ent = ent.unsqueeze(dim=1).repeat(1, B, 1).sum(-1)
    ce = torch.matmul(p, q.log().T)
    kl = ent - ce
    return kl

def get_jsd_matrix(p, q, r):
    m = (p + q + r) / 3
    p, q, r = torch.clamp(p, min=1e-7, max=1), \
              torch.clamp(q, min=1e-7, max=1), \
              torch.clamp(r, min=1e-7, max=1)
    jsd = (get_kl_matrix(p, m) + get_kl_matrix(q, m) + get_kl_matrix(r, m)) / 3
    return jsd


def get_jsd_matrix_inv(p, q, r):
    m = (p + q + r) / 3
    jsd = (get_kl_matrix(m, p) + get_kl_matrix(m, p) + get_kl_matrix(m, r)) / 3
    return jsd


def get_jsd_matrix_2arg(p, q):
    m = (p + q ) / 2
    p, q = torch.clamp(p, min=1e-7, max=1), \
           torch.clamp(q, min=1e-7, max=1)
    jsd = (get_kl_matrix(p, m) + get_kl_matrix(q, m)) / 2
    return jsd


def make_matrix(p, q, criterion, reduction='none'):
    '''
    Input:
        p: (B,C)
        q: (B,C)
        criterion:  {class} reduction='none' is recommended.
        reduction:  {str} Reduction you want.
    By dshong
    '''
    if len(p.shape) == 1:
        p = p.unsqueeze(1)
    if len(q.shape) == 1:
        q = q.unsqueeze(1)

    assert p.size() == q.size()
    B, C = p.size()

    # 1. Expand the dimension:
    #    p: (B,C) →[unsqueeze]→ (B,1,C) →[repeat]→ (B,B,C) →[reshape]→ (B*B,C)
    #    q: (B,C) →[unsqueeze]→ (1,B,C) →[repeat]→ (B,B,C) →[reshape]→ (B*B,C)
    p_exp = p.unsqueeze(1).repeat(1, B, 1).reshape(-1, C)
    q_exp = q.unsqueeze(0).repeat(B, 1, 1).reshape(-1, C)

    # 2. Compute with criterion:
    matrix = criterion(q_exp, p_exp)

    # 3. Do reduction
    if reduction == 'none':
        matrix = matrix.reshape(B, B, -1)
    elif reduction == 'mean':
        matrix = torch.mean(matrix, dim=-1)
        matrix = matrix.reshape(B, B)
    elif reduction == 'sum':
        matrix = torch.sum(matrix, dim=-1)
        matrix = matrix.reshape(B, B)
    else:
        raise ValueError(f'unexpected reduction type {reduction}.')

    return matrix


def jsdv3(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = temper * temper * jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    # jsd_distance2 = lambda_weight * jsd_distance
    # jsd_value = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = lambda_weight * jsd_distance

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv4(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss
    version 5
    matrix with nn.module loss function by dazory
    detach sum function to break gradient
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]


    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    # jsd_distance2 = lambda_weight * jsd_distance
    # jsd_value = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    loss = lambda_weight * jsd_distance

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                # 'jsd_matrix': jsd_matrix,
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                # 'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_test(logits_clean, logits_aug1, logits_aug2=None, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, = F.softmax(logits_clean / temper, dim=1), \
                       F.softmax(logits_aug1 / temper, dim=1)

    jsd_matrix = get_jsd_matrix_2arg(p_clean, p_aug1)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    # jsd_distance2 = lambda_weight * jsd_distance
    # jsd_value = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                '_aug2': p_aug2,
                }

    return loss, features


def jsdv3_cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    similarity = (F.cosine_similarity(logits_clean, logits_aug1) +
                  F.cosine_similarity(logits_clean, logits_aug2) +
                  F.cosine_similarity(logits_aug1, logits_aug2)) / 3
    similarity = similarity.mean()

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    # jsd_distance2 = lambda_weight * jsd_distance
    # jsd_value = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = -1 * lambda_weight * similarity

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'similarity': similarity,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                }

    return loss, features


def jsdv3_msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    with constrastive MSE loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    # MSE loss

    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug1, logits_aug2, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug2, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum() / mask_same_triuu.sum()

    mse_matrix_same_class_max, _ = mse_matrix_same_class.max(dim=1, keepdim=True)
    mse_matrix_same_class_max = mse_matrix_same_class_max.repeat((1, batch_size)) * mask_diff_triuu
    mse_matrix_triplet = torch.clamp((mse_matrix_same_class_max - mse_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = mse_matrix_triplet.sum() / (torch.count_nonzero(mse_matrix_triplet) + 1e-7)

    # JSD loss
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = lambda_weight * triplet_loss

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                'mse_distance': mse_distance,
                'mse_distance_same_class': mse_distance_same_class,
                'mse_distance_diff_class': mse_distance_diff_class,
                'mse_matrix': mse_matrix,
                'triplet_loss': triplet_loss,
                }

    return loss, features


def jsdv3_msev1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    with additional MSE loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_diff_triuu = mask_diff_class * mask_triuu
    mask_same_triuu = mask_same_class * mask_triuu


    # MSE loss

    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug1, logits_aug2, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug2, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum() / mask_same_instance.sum()



    # JSD loss
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = jsd_distance + lambda_weight * mse_distance

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                'mse_distance': mse_distance,
                'mse_matrix': mse_matrix,
                }

    return loss, features



def jsdv3_msev1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    with constrastive MSE loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    # MSE loss

    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug1, logits_aug2, criterion=nn.MSELoss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug2, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='sum')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum() / mask_same_triuu.sum()

    mse_matrix_same_class_max, _ = mse_matrix_same_class.max(dim=1, keepdim=True)
    mse_matrix_same_class_max = mse_matrix_same_class_max.repeat((1, batch_size)) * mask_diff_triuu
    mse_matrix_triplet = torch.clamp((mse_matrix_same_class_max.detach() - mse_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = mse_matrix_triplet.sum() / (torch.count_nonzero(mse_matrix_triplet) + 1e-7)

    # JSD loss
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = lambda_weight * triplet_loss

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                'mse_distance': mse_distance,
                'mse_distance_same_class': mse_distance_same_class,
                'mse_distance_diff_class': mse_distance_diff_class,
                'mse_matrix': mse_matrix,
                'triplet_loss': triplet_loss,
                }

    return loss, features


def jsdv3_l1v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    with constrastive L1 loss
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    # L1 loss

    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.L1Loss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug1, logits_aug2, criterion=nn.L1Loss(reduction='none'), reduction='sum') + \
                 make_matrix(logits_aug2, logits_aug1, criterion=nn.L1Loss(reduction='none'), reduction='sum')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum() / mask_same_triuu.sum()

    mse_matrix_same_class_max, _ = mse_matrix_same_class.max(dim=1, keepdim=True)
    mse_matrix_same_class_max = mse_matrix_same_class_max.repeat((1, batch_size)) * mask_diff_triuu
    mse_matrix_triplet = torch.clamp((mse_matrix_same_class_max - mse_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = mse_matrix_triplet.sum() / (torch.count_nonzero(mse_matrix_triplet) + 1e-7)

    # JSD loss
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = lambda_weight * triplet_loss

    # jsd_debug = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=lambda_weight)

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                'mse_distance': mse_distance,
                'mse_distance_same_class': mse_distance_same_class,
                'mse_distance_diff_class': mse_distance_diff_class,
                'mse_matrix': mse_matrix,
                'triplet_loss': triplet_loss,
                }

    return loss, features



def jsdv3_0_1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3

    triplet jsd loss between positives (same class)  and negatives (diff class) between p_orig and p_aug
    hard positive max version
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    jsd_matrix_same_class_max, _ = jsd_matrix_same_class.max(dim=1, keepdim=True)
    jsd_matrix_same_class_max = jsd_matrix_same_class_max.repeat((1, batch_size)) * mask_diff_triuu
    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_class_max - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_0_2(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3
    triplet jsd loss between positives (same class)  and negatives (diff class) between p_orig and p_aug
    positive mean version
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    count = (torch.count_nonzero(jsd_matrix_same_class, dim=1) + 1e-7)
    count = count.unsqueeze(dim=1)
    jsd_matrix_same_class_mean = jsd_matrix_same_class.sum(dim=1, keepdim=True) / count
    jsd_matrix_same_class_mean = jsd_matrix_same_class_mean.repeat((1, batch_size)) * mask_diff_triuu

    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_class_mean - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_0_3(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3.0.1
    same class -> same instance

    triplet jsd loss between positives (same instance)  and negatives (diff class) between p_orig and p_aug
    hard positive max version

    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    jsd_matrix_same_instance_max, _ = jsd_matrix_same_instance.max(dim=1, keepdim=True)
    jsd_matrix_same_instance_max = jsd_matrix_same_instance_max.repeat((1, batch_size)) * mask_diff_triuu
    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_instance_max - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_0_1_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3

    triplet jsd loss between positives (same class)  and negatives (diff class) between p_orig and p_aug
    hard positive max version
    detach the same class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    jsd_matrix_same_class_max, _ = jsd_matrix_same_class.max(dim=1, keepdim=True)
    jsd_matrix_same_class_max = jsd_matrix_same_class_max.repeat((1, batch_size)) * mask_diff_triuu
    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_class_max.detach() - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_0_2_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3
    triplet jsd loss between positives (same class)  and negatives (diff class) between p_orig and p_aug
    positive mean version
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    count = (torch.count_nonzero(jsd_matrix_same_class, dim=1) + 1e-7)
    count = count.unsqueeze(dim=1)
    jsd_matrix_same_class_mean = jsd_matrix_same_class.sum(dim=1, keepdim=True) / count
    jsd_matrix_same_class_mean = jsd_matrix_same_class_mean.repeat((1, batch_size)) * mask_diff_triuu

    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_class_mean.detach() - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_0_3_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    edited from jsdv3.0.1
    same class -> same instance

    triplet jsd loss between positives (same instance)  and negatives (diff class) between p_orig and p_aug
    hard positive max version

    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance

    jsd_matrix_same_instance_max, _ = jsd_matrix_same_instance.max(dim=1, keepdim=True)
    jsd_matrix_same_instance_max = jsd_matrix_same_instance_max.repeat((1, batch_size)) * mask_diff_triuu
    jsd_matrix_triplet = torch.clamp((jsd_matrix_same_instance_max.detach() - jsd_matrix_diff_class + margin) * mask_diff_triuu, min=0)
    triplet_loss = jsd_matrix_triplet.sum() / (torch.count_nonzero(jsd_matrix_triplet) + 1e-7)

    loss += lambda_weight * triplet_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_log_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss. Augmix original version
    log JSD matrix and JSD matrix_inv
    edited from jsdv3
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_inv = get_jsd_matrix_inv(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
    jsd_matrix_same_instance_inv = jsd_matrix_inv * mask_same_instance
    jsd_distance_inv = jsd_matrix_same_instance_inv.sum() / mask_same_instance.sum()


    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
    jsd_matrix_diff_class_inv = jsd_matrix_inv * mask_diff_triuu
    jsd_distance_diff_class_inv = jsd_matrix_diff_class_inv.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
    jsd_matrix_same_class_inv = jsd_matrix_inv * mask_same_triuu
    jsd_distance_same_class_inv = jsd_matrix_same_class_inv.sum() / mask_same_triuu.sum()

    loss = lambda_weight * jsd_distance

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_inv': jsd_distance_inv,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_diff_class_inv': jsd_distance_diff_class_inv,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_distance_same_class_inv': jsd_distance_same_class_inv,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


def jsdv3_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None):
    '''
    JSD matrix loss.

    log JSD matrix and JSD matrix_inv
    edited from jsdv3_log_inv
    optimize JSD inversion
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
    jsd_matrix_inv = get_jsd_matrix_inv(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
    jsd_matrix_same_instance_inv = jsd_matrix_inv * mask_same_instance
    jsd_distance_inv = jsd_matrix_same_instance_inv.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
    jsd_matrix_diff_class_inv = jsd_matrix_inv * mask_diff_triuu
    jsd_distance_diff_class_inv = jsd_matrix_diff_class_inv.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
    jsd_matrix_same_class_inv = jsd_matrix_inv * mask_same_triuu
    jsd_distance_same_class_inv = jsd_matrix_same_class_inv.sum() / mask_same_triuu.sum()

    loss = lambda_weight * jsd_distance_inv

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_inv': jsd_distance_inv,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_diff_class_inv': jsd_distance_diff_class_inv,
                'jsd_distance_same_class': jsd_distance_same_class,
                'jsd_distance_same_class_inv': jsd_distance_same_class_inv,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features


# def jsdv3_03(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
#     '''
#     JSD matrix loss
#     diff maximize
#     '''
#
#     device = logits_clean.device
#     pred_clean = logits_clean.data.max(1)[1]
#     pred_aug1 = logits_aug1.data.max(1)[1]
#     pred_aug2 = logits_aug2.data.max(1)[1]
#
#     batch_size = logits_clean.size()[0]
#     targets = targets.contiguous().view(-1, 1)  # [B, 1]
#     temper = 1.0
#
#     mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
#     mask_triu = torch.triu(mask_identical.clone().detach())
#     mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
#     mask_triuu = mask_triu - mask_same_instance
#     mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
#     mask_diff_class = 1 - mask_same_class  # [B, B]
#     p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
#                              F.softmax(logits_aug2 / temper, dim=1)
# #
# #     jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
# #
# #     jsd_matrix_same_instance = jsd_matrix * mask_same_instance
# #     jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
# #
# #     mask_diff_triuu = mask_diff_class * mask_triuu
# #     jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
# #     jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
# #
# #     mask_same_triuu = mask_same_class * mask_triuu
# #     jsd_matrix_same_class = jsd_matrix * mask_same_triuu
# #     jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
# #
# #     loss = 12 * jsd_distance - lambda_weight * jsd_distance_diff_class
# #     triplet_loss = 0
# #
# #     features = {'jsd_distance': jsd_distance,
# #                 'jsd_distance_diff_class': jsd_distance_diff_class,
# #                 'jsd_distance_same_class': jsd_distance_same_class,
# #                 'triplet_loss': triplet_loss,
# #                 'jsd_matrix': jsd_matrix,
# #                 'p_clean': p_clean,
# #                 'p_aug1': p_aug1,
# #                 }
# #
# #     return loss, features
# #
# #
# # def jsdv3_04(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
# #     '''
# #     JSD matrix loss
# #     same minimize
# #     '''
# #
# #     device = logits_clean.device
# #     pred_clean = logits_clean.data.max(1)[1]
# #     pred_aug1 = logits_aug1.data.max(1)[1]
# #     pred_aug2 = logits_aug2.data.max(1)[1]
# #
# #     batch_size = logits_clean.size()[0]
# #     targets = targets.contiguous().view(-1, 1)  # [B, 1]
# #     temper = 1.0
# #
# #     mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
# #     mask_triu = torch.triu(mask_identical.clone().detach())
# #     mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
# #     mask_triuu = mask_triu - mask_same_instance
# #     mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
# #     mask_diff_class = 1 - mask_same_class  # [B, B]
# #     p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
# #                               F.softmax(logits_aug1 / temper, dim=1), \
# #                               F.softmax(logits_aug2 / temper, dim=1)
# #
# #     jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
# #
# #     jsd_matrix_same_instance = jsd_matrix * mask_same_instance
# #     jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
# #
# #     mask_diff_triuu = mask_diff_class * mask_triuu
# #     jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
# #     jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
# #
# #     mask_same_triuu = mask_same_class * mask_triuu
# #     jsd_matrix_same_class = jsd_matrix * mask_same_triuu
# #     jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
# #
# #     loss = 12 * jsd_distance + lambda_weight * jsd_distance_same_class
# #
# #     triplet_loss = 0
# #
# #     features = {'jsd_distance': jsd_distance,
# #                 'jsd_distance_diff_class': jsd_distance_diff_class,
# #                 'jsd_distance_same_class': jsd_distance_same_class,
# #                 'triplet_loss': triplet_loss,
# #                 'jsd_matrix': jsd_matrix,
# #                 'p_clean': p_clean,
# #                 'p_aug1': p_aug1,
# #                 }
# #
# #     return loss, features                  F.softmax(logits_aug1 / temper, dim=1), \
#


def jsd_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance, cossim same class and diff class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrast(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.


    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance_loss + lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance_loss': jsd_distance_loss,
                'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                # 'jsd_matrix': jsd_matrix,
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                }

    return loss, features


def jsd_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance,
    SupContrastv0_01: same class positives, diff class negatives
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance_loss + lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance_loss': jsd_distance_loss,
                'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                # 'jsd_matrix': jsd_matrix,
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                }

    return loss, features


def jsd_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance,
    SupContrastv0_01: same class positives, diff class negatives
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance_loss + lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance_loss': jsd_distance_loss,
                'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                # 'jsd_matrix': jsd_matrix,
                # 'p_clean': p_clean,
                # 'p_aug1': p_aug1,
                }

    return loss, features



def jsdv3_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance, cossim same class and diff class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrast(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance + ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                }

    return loss, features


def jsdv3_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance,
    SupContrastv0_01: same class positives, diff class negatives
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance + ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                }

    return loss, features


def jsdv3_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance,
    SupContrastv0_01: same class positives, diff class negatives
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance + ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                }

    return loss, features


def jsdv3_ntxent_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance, cossim same class and diff class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrast_detach(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()

    loss = 12 * jsd_distance + ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance,
                'jsd_distance_diff_class': jsd_distance_diff_class,
                'jsd_distance_same_class': jsd_distance_same_class,
                'triplet_loss': triplet_loss,
                'jsd_matrix': jsd_matrix,
                'p_clean': p_clean,
                'p_aug1': p_aug1,
                }

    return loss, features


def jsd_kd(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, reduction='batchmean'):

    logits_clean = logits_clean.detach()
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction=reduction) +
                            F.kl_div(p_mixture, p_aug1, reduction=reduction) +
                            F.kl_div(p_mixture, p_aug2, reduction=reduction)) / 3.

    return loss


def jsd_kdv2(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, reduction='batchmean'):
    logits_clean = logits_clean.detach()
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3
    loss = lambda_weight * (F.kl_div(p_clean.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug1.log(), p_mixture, reduction='batchmean') +
                            F.kl_div(p_aug2.log(), p_mixture, reduction='batchmean')) / 3.

    return loss


def jsd_distance(logit1, logit2, reduction='batchmean'):
    p1, p2 = F.softmax(logit1, dim=1), F.softmax(logit2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p1 + p2) / 2., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p1, reduction=reduction) + F.kl_div(p_mixture, p2, reduction=reduction)) / 2.

    return loss


def jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=0.5, reduction='batchmean'):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction=reduction) +
                            F.kl_div(p_mixture, p_aug1, reduction=reduction) +
                            F.kl_div(p_mixture, p_aug2, reduction=reduction)) / 3.

    return loss


def kl_v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug2, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 6.

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    features = {'jsd_distance': jsd_distance.detach(),
                }

    return loss, features


def kl_v1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 4.

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    features = {'jsd_distance': jsd_distance.detach(),
                }

    return loss, features


def kl_v1_2(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean')) / 2.

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_distance = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    features = {'jsd_distance': jsd_distance.detach(),
                }

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



def kl_v1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean = p_clean.detach()
    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug2, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 6.

    return loss


def kl_v1_1_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)
    p_clean = p_clean.detach()

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 4.
    return loss


def kl_v1_2_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean = p_clean.detach()
    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean')) / 2.

    return loss

def kl_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), \
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug2, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_clean, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 6.

    return loss


def kl_v1_1_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_clean_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug2, reduction='batchmean') +
                            F.kl_div(p_aug2_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_aug1_log, p_aug2, reduction='batchmean')) / 4.
    return loss



def kl_v1_2_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    p_clean_log = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1_log = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2_log = torch.clamp(p_aug2, 1e-7, 1).log()

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_clean_log, p_aug1, reduction='batchmean') +
                            F.kl_div(p_clean_log, p_aug2, reduction='batchmean')) / 2.

    return loss


def msev1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    # collapse with lw 12
    B, C = logits_clean.size()

    distance = (F.mse_loss(logits_clean, logits_aug1, reduction='sum') +
            F.mse_loss(logits_clean, logits_aug2, reduction='sum') +
            F.mse_loss(logits_aug1, logits_aug2, reduction='sum')) / 3 / B

    loss = lambda_weight * distance
    feature = {'distance': distance.detach()}

    return loss, feature

def msev1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    # collapse with lw 12
    B, C = logits_clean.size()

    distance = (F.mse_loss(logits_clean, logits_aug1, reduction='mean') +
            F.mse_loss(logits_clean, logits_aug2, reduction='mean') +
            F.mse_loss(logits_aug1, logits_aug2, reduction='mean')) / 3

    loss = lambda_weight * distance
    feature = {'distance': distance.detach()}

    return loss, feature

def mse_v1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):

    B, C = logits_clean.size()
    logits_clean = logits_clean.detach()
    loss = (F.mse_loss(logits_clean, logits_aug1, reduction='sum') +
            F.mse_loss(logits_clean, logits_aug2, reduction='sum') +
            F.mse_loss(logits_aug1, logits_aug2, reduction='sum')) / 3 / B

    loss = lambda_weight * loss

    return loss


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

def ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets, temper=1):

    sim1 = F.cosine_similarity(logits_clean, logits_aug1)
    sim2 = F.cosine_similarity(logits_aug1, logits_aug2)
    sim3 = F.cosine_similarity(logits_aug2, logits_clean)
    loss = sim1 / temper

    jsd1 = jsd_distance(logits_clean, logits_aug1, 'batchmean')
    jsd2 = jsd_distance(logits_aug1, logits_aug2, 'batchmean')
    jsd3 = jsd_distance(logits_aug2, logits_clean, 'batchmean')
    # sorted_targets, indices = torch.sort(targets)
    # sorted_logits_clean = logits_clean[indices]
    logits_clean_0 = logits_clean[targets==0]
    return loss


def supcontrast(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
    original supcontrast loss
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device

    # temporary deprecated
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),


    logits_clean, logits_aug1, logits_aug2 = torch.unsqueeze(logits_clean, dim=1), \
                                             torch.unsqueeze(logits_aug1, dim=1), \
                                             torch.unsqueeze(logits_aug2, dim=1)
    features = torch.cat([logits_clean, logits_aug1, logits_aug2], dim=1)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temper)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def supcontrast_detach(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
    original supcontrast loss
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device

    # temporary deprecated
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),


    logits_clean, logits_aug1, logits_aug2 = torch.unsqueeze(logits_clean, dim=1), \
                                             torch.unsqueeze(logits_aug1, dim=1), \
                                             torch.unsqueeze(logits_aug2, dim=1)
    features = torch.cat([logits_clean, logits_aug1, logits_aug2], dim=1)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temper)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits.detach() - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def supcontrast_test(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
    original supcontrast loss
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device

    # temporary deprecated
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),


    logits_clean, logits_aug1, logits_aug2 = torch.unsqueeze(logits_clean, dim=1), \
                                             torch.unsqueeze(logits_aug1, dim=1), \
                                             torch.unsqueeze(logits_aug2, dim=1)
    features = torch.cat([logits_clean, logits_aug1, logits_aug2], dim=1)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)
    mask_np = mask.cpu().detach().numpy()   # same instance or same class
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temper)
    anchor_dot_contrast_np = anchor_dot_contrast.cpu().detach().numpy()
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_np = logits.cpu().detach().numpy()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    mask_np2 = mask.cpu().detach().numpy()  # self case, same instance, same class for augmix
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask_np3 = logits_mask.cpu().detach().numpy()   # 1 - self case
    mask = mask * logits_mask
    mask_np4 = mask.cpu().detach().numpy()  # same instance, same class for augmix

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    exp_logits_np = exp_logits.cpu().detach().numpy() # denominator of ntxent
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    log_exp_logits_sum_np = torch.log(exp_logits.sum(1, keepdim=True)).cpu().detach().numpy()
    log_prob_np = log_prob.cpu().detach().numpy()

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # mask: self instance, self contrast case
    mean_log_prob_pos_np = mean_log_prob_pos.cpu().detach().numpy()

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def supcontrast_maskv0_01(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):
    base_temper = temper

    logits_anchor, logits_contrast = F.normalize(logits_anchor, dim=1), F.normalize(logits_contrast, dim=1)

    anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits) * mask_contrast
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / mask_anchor.sum(1)
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
    supcontrast loss
    same instance, class: positive, diff class: negative, same class: skip
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    targets = targets.contiguous().view(-1, 1)
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_same_class_np = mask_same_class.detach().cpu().numpy()
    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)


    # mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_aug1, targets, mask_same_class, mask_identical, lambda_weight, temper)
    loss2 = supcontrast_maskv0_01(logits_clean, logits_aug2, targets, mask_same_class, mask_identical, lambda_weight, temper)
    loss3 = supcontrast_maskv0_01(logits_aug1, logits_aug2, targets, mask_same_class, mask_identical, lambda_weight, temper)

    loss = (loss1 + loss2 + loss3) / 3

    return loss


def supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
    supcontrast loss
    same instance: positive, diff class: negative, same class: skip
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    # temporary deprecated


    targets = targets.contiguous().view(-1, 1)
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    # mask_same_class_np = mask_same_class.detach().cpu().numpy()
    mask_diff_class = 1 - mask_same_class  # [B, B]
    # mask_diff_class_np = mask_diff_class.detach().cpu().numpy()
    mask_same_instance_diff_class = mask_same_instance + mask_diff_class
    # mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_aug1, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss2 = supcontrast_maskv0_01(logits_clean, logits_aug2, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss3 = supcontrast_maskv0_01(logits_aug1, logits_aug2, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)

    loss = (loss1 + loss2 + loss3) / 3

    return loss


def supconv0_01_norm(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=1, temper=1.0,
                     reduction='batchmean'):
    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device

    # temporary deprecated
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),

    logits_clean, logits_aug1, logits_aug2 = torch.unsqueeze(logits_clean, dim=1), \
                                             torch.unsqueeze(logits_aug1, dim=1), \
                                             torch.unsqueeze(logits_aug2, dim=1)
    features = torch.cat([logits_clean, logits_aug1, logits_aug2], dim=1)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
    anchor_feature_norm = torch.norm(anchor_feature, p=2, dim=1, keepdim=True)
    anchor_dot_norm = torch.matmul(anchor_feature_norm, anchor_feature_norm.T)

    logits = torch.div(anchor_dot_contrast, anchor_dot_norm.detach())
    logits = torch.div(logits, temper)

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

def jsdv4_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    logging jsd distance by jsdv4
    supcontrast function
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrast(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    loss = lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance.detach(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'triplet_loss': triplet_loss,
                }

    return loss, features


def jsdv4_ntxentv0_01(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    logging jsd distance by jsdv4
    supcontrastv0.01 function
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    loss = lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance.detach(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'triplet_loss': triplet_loss,
                }

    return loss, features


def jsdv4_ntxentv0_02(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    logging jsd distance by jsdv4
    supcontrastv0.01 function
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    loss = lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance.detach(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'triplet_loss': triplet_loss,
                }

    return loss, features


def jsdv4_ntxent_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    logging jsd distance by jsdv4
    supcontrast function
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supcontrast_detach(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]

    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    mask_diff_triuu = mask_diff_class * mask_triuu
    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mask_same_triuu = mask_same_class * mask_triuu
    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    loss = lambda_weight * ntxent_loss
    triplet_loss = ntxent_loss

    features = {'jsd_distance': jsd_distance.detach(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'triplet_loss': triplet_loss,
                }

    return loss, features


def opl(features_clean, features_aug1, features_aug2, args, lambda_weight=12, temper=1.0, labels=None):
    """
    norm true
    use only features_clean
    w/o attention
    """
    norm = args.opl_norm
    attention = args.opl_attention
    gamma = args.opl_gamma

    features = features_clean
    device = features_clean.device

    if attention:
        features_weights = torch.matmul(features, features.T)
        features_weights = F.softmax(features_weights, dim=1)
        features = torch.matmul(features_weights, features)

        #  features are normalized
    if norm:
        features = F.normalize(features, p=2, dim=1)

    labels = labels[:, None]  # extend dim
    mask = torch.eq(labels, labels.t()).bool().to(device)
    eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features, features.t())

    pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
    neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

    opl_loss = (1.0 - pos_pairs_mean) + (gamma * neg_pairs_mean)
    loss = lambda_weight * opl_loss
    # loss = neg_pairs_mean

    features = {'opl_loss': opl_loss.detach(),
                'pos_pairs_mean': pos_pairs_mean.detach(),
                'neg_pairs_mean': neg_pairs_mean.detach(),
    }

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


def _ssim(img1, img2, window, window_size, channel, size_average=True):
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
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def ssim(img1, img2, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim(args, img_clean, img_aug1, img_aug2, lambda_weight, targets, temper=1, reduction='mean'):

    window_size = args.window
    if reduction == 'mean':
        size_average = True
    (_, channel, _, _) = img_clean.size()
    window = create_window(window_size, channel)
    if img_clean.is_cuda:
        window = window.cuda(img_clean.get_device())
    window = window.type_as(img_clean)

    loss1 = _ssim(img_clean, img_aug1, window, window_size, channel, size_average)
    loss2 = _ssim(img_aug1, img_aug2, window, window_size, channel, size_average)
    loss3 = _ssim(img_aug2, img_clean, window, window_size, channel, size_average)

    loss = 1 - (loss1 + loss2 + loss3) / 3
    features = {'distance': loss.detach()}
    loss = loss * lambda_weight

    return loss, features