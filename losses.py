import torch
import torch.nn.functional as F


def get_additional_loss(name, logits_clean, logits_aug1, logits_aug2, lambda_weight=12, **kwargs):

    if name == 'none':
        loss = 0
    elif name == 'jsd':
        loss = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsd_temper':
        loss = jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'kl':
        loss = kl(logits_clean, logits_aug1, logits_aug2, lambda_weight)

    return loss

def jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return loss


def jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=2):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1),\
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return loss

def kl(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), \
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    loss = lambda_weight * (F.kl_div(p_aug1, p_clean, reduction='batchmean') +
                            F.kl_div(p_clean, p_aug2, reduction='batchmean') +
                            F.kl_div(p_aug2, p_aug1, reduction='batchmean')) / 3.

    return loss

def ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    torch.dot(logits_clean, logits_aug1)