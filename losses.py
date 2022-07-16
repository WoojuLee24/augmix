import torch
import torch.nn.functional as F

def jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight=12):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                              F.softmax(logits_aug1, dim=1), \
                              F.softmax(logits_aug2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    jsd_loss = lambda_weight * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return jsd_loss