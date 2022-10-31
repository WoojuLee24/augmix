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
    elif name == 'jsdv3.test':
        loss, features = jsdv3_test(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.cossim':
        loss, features = jsdv3_cossim(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
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

    elif name == 'jsdv3.log.inv':
        loss, features = jsdv3_log_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features
    elif name == 'jsdv3.inv':
        loss, features = jsdv3_inv(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features

        return loss, features
    # elif name == 'jsdv3.04':
    #     margin = args.margin
    #     loss, features = jsdv3_04(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
    #     return loss, features
    elif name == 'jsdv3.ntxent':
        margin = args.margin
        loss, features = jsdv3_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
        return loss, features
    elif name == 'jsdv3.ntxent.diff':
        margin = args.margin
        loss, features = jsdv3_ntxent_diff(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets, margin)
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
    elif name == 'supconv0.01':
        loss = supconv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight,  temper, reduction)
    elif name == 'supconv0.01_test':
        loss = supconv0_01_test(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction)
    elif name == 'supconv0.01.diff':
        loss = supconv0_01_diff(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction)
    elif name == 'supconv0.02':
        loss = supconv0_02(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight,  temper, reduction)
    elif name == 'jsdv3_apr_p':
        loss, features = jsdv3(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper, targets)
        return loss, features

    # kl_div
    elif name == 'klv1.0':
        loss = kl_v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.1':
        loss = kl_v1_1(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'klv1.2':
        loss = kl_v1_2(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
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
        loss = mse_v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'msev1.0.detach':
        loss = mse_v1_0_detach(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)

    # analysis test mode
    elif name == 'analysisv1.0':
        loss, features = analysisv1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight)
        return loss, features

    return loss

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
    triplet_loss = jsd_matrix_triplet.sum() / torch.count_nonzero(jsd_matrix_triplet)

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
    triplet_loss = jsd_matrix_triplet.sum() / torch.count_nonzero(jsd_matrix_triplet)

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
    triplet_loss = jsd_matrix_triplet.sum() / torch.count_nonzero(jsd_matrix_triplet)

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
    triplet_loss = jsd_matrix_triplet.sum() / torch.count_nonzero(jsd_matrix_triplet)

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
#                               F.softmax(logits_aug1 / temper, dim=1), \
#                               F.softmax(logits_aug2 / temper, dim=1)
#
#     jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
#
#     jsd_matrix_same_instance = jsd_matrix * mask_same_instance
#     jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
#
#     mask_diff_triuu = mask_diff_class * mask_triuu
#     jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
#     jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
#
#     mask_same_triuu = mask_same_class * mask_triuu
#     jsd_matrix_same_class = jsd_matrix * mask_same_triuu
#     jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
#
#     loss = 12 * jsd_distance - lambda_weight * jsd_distance_diff_class
#     triplet_loss = 0
#
#     features = {'jsd_distance': jsd_distance,
#                 'jsd_distance_diff_class': jsd_distance_diff_class,
#                 'jsd_distance_same_class': jsd_distance_same_class,
#                 'triplet_loss': triplet_loss,
#                 'jsd_matrix': jsd_matrix,
#                 'p_clean': p_clean,
#                 'p_aug1': p_aug1,
#                 }
#
#     return loss, features
#
#
# def jsdv3_04(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
#     '''
#     JSD matrix loss
#     same minimize
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
#                               F.softmax(logits_aug1 / temper, dim=1), \
#                               F.softmax(logits_aug2 / temper, dim=1)
#
#     jsd_matrix = get_jsd_matrix(p_clean, p_aug1, p_aug2)
#
#     jsd_matrix_same_instance = jsd_matrix * mask_same_instance
#     jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum()
#
#     mask_diff_triuu = mask_diff_class * mask_triuu
#     jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
#     jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum()
#
#     mask_same_triuu = mask_same_class * mask_triuu
#     jsd_matrix_same_class = jsd_matrix * mask_same_triuu
#     jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum()
#
#     loss = 12 * jsd_distance + lambda_weight * jsd_distance_same_class
#
#     triplet_loss = 0
#
#     features = {'jsd_distance': jsd_distance,
#                 'jsd_distance_diff_class': jsd_distance_diff_class,
#                 'jsd_distance_same_class': jsd_distance_same_class,
#                 'triplet_loss': triplet_loss,
#                 'jsd_matrix': jsd_matrix,
#                 'p_clean': p_clean,
#                 'p_aug1': p_aug1,
#                 }
#
#     return loss, features


def jsdv3_ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance, cossim same class and diff class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supconv0_01(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

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


def jsdv3_ntxent_diff(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0, targets=None, margin=0.02):
    '''
    JSD matrix loss
    JSD same instance, cossim same class and diff class
    '''

    device = logits_clean.device
    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    ntxent_loss = supconv0_01_diff(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

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

    return loss


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
    return loss


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
    return loss


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


def mse_v1_0(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=1.0):
    # collapse with lw 12
    B, C = logits_clean.size()

    loss = (F.mse_loss(logits_clean, logits_aug1, reduction='sum') +
            F.mse_loss(logits_clean, logits_aug2, reduction='sum') +
            F.mse_loss(logits_aug1, logits_aug2, reduction='sum')) / 3 / B

    loss = lambda_weight * loss

    return loss


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
    logits = (sim1 + sim2 + sim3) / 3
    loss = - logits.mean() / temper * lambda_weight

    # jsd1 = jsd_distance(logits_clean, logits_aug1, 'batchmean')
    # jsd2 = jsd_distance(logits_aug1, logits_aug2, 'batchmean')
    # jsd3 = jsd_distance(logits_aug2, logits_clean, 'batchmean')
    # logits2 = (jsd1 + jsd2 + jsd3) / 3
    # logits2 = logits2 / temper

    return loss

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

def supconv0_01(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

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
    loss *= lambda_weight

    return loss


def supconv0_01_test(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

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
    mask_np = mask.cpu().detach().numpy()
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
    mask_np2 = mask.cpu().detach().numpy()
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask_np3 = logits_mask.cpu().detach().numpy()
    mask = mask * logits_mask
    mask_np4 = mask.cpu().detach().numpy()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    exp_logits_np = exp_logits.cpu().detach().numpy()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    log_exp_logits_sum_np = torch.log(exp_logits.sum(1, keepdim=True)).cpu().detach().numpy()
    log_prob_np = log_prob.cpu().detach().numpy()

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    mean_log_prob_pos_np = mean_log_prob_pos.cpu().detach().numpy()

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    loss *= lambda_weight

    return loss


def supcon_maskv0_01(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):
    base_temper = temper

    anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits) * mask_contrast
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / mask_contrast.sum(1)
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.mean()
    return loss


def supconv0_01_diff(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

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
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),


    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_same_instance_diff_class = mask_same_instance + mask_diff_class

    loss1 = supcon_maskv0_01(logits_clean, logits_aug1, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss2 = supcon_maskv0_01(logits_clean, logits_aug2, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss = lambda_weight * (loss1 + loss2)

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
    loss *= lambda_weight

    return loss


####

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = torch.cat([labels, labels, labels], dim=0)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size * 1 / 64

        return loss


class MlpJSDLoss(nn.Module):
    """Mlp JSD loss.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, in_feature, out_feature):
        super(MlpJSDLoss, self).__init__()
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, args, logits_all=None, features=None, targets=None, split=3, lambda_weight=12):
        if args.jsd_layer == 'logits':
            embedded = self.fc(logits_all)
        elif args.jsd_layer == 'features':
            embedded = self.fc(features)
        logits_clean, logits_aug1, logits_aug2 = torch.chunk(embedded, split)

        loss = jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight, args.temper)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

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
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss