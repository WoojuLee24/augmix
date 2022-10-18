import torch
import torch.nn as nn
import torch.nn.functional as F


# Jensen-Shannon Divergence
def jsd_with_key(outputs1, outputs2, outputs3, key='prediction'):
    p_clean, p_aug1, p_aug2 = F.softmax(outputs1[key], dim=1), \
                              F.softmax(outputs2[key], dim=1), \
                              F.softmax(outputs3[key], dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    features = {'p_clean': p_clean,
                'p_aug1': p_aug1,
                'p_aug2': p_aug2,
                }

    return loss, features

def jsd_pred(outputs1, outputs2, outputs3):
    return jsd_with_key(outputs1, outputs2, outputs3, key='prediction')

def jsd_proj(outputs1, outputs2, outputs3):
    return jsd_with_key(outputs1, outputs2, outputs3, key='projection')

def jsd_repr(outputs1, outputs2, outputs3):
    return jsd_with_key(outputs1, outputs2, outputs3, key='representation')

def jsd(outputs1, outputs2, outputs3):
    pred = [F.softmax(outputs1['prediction'], dim=1),
            F.softmax(outputs2['prediction'], dim=1),
            F.softmax(outputs3['prediction'], dim=1)]
    proj = [F.softmax(outputs1['projection'], dim=1),
            F.softmax(outputs2['projection'], dim=1),
            F.softmax(outputs3['projection'], dim=1)]

    losses = torch.zeros(len(pred), len(proj))
    for i in range(len(pred)):
        for j in range(len(proj)):
            if i == j:
                continue
            p_mixture = torch.clamp((pred[i] + proj[j]) / 2., 1e-7, 1).log()
            losses[i, j] = (F.kl_div(p_mixture.detach(), pred[i], reduction='batchmean') +
                            F.kl_div(p_mixture.detach(), proj[j], reduction='batchmean')) / 2.

    loss = torch.sum(losses) / 6.

    features = {'p_clean': pred[0],
                'p_aug1': pred[1],
                'p_aug2': pred[2],
                }

    return loss, features

# Cosine similarity
def cosine_similarity(outputs1, outputs2, outputs3):
    criterion = nn.CosineSimilarity(dim=1)

    pred1, pred2, pred3 = outputs1['prediction'], outputs2['prediction'], outputs3['prediction']
    proj1, proj2, proj3 = outputs1['projection'], outputs2['projection'], outputs3['projection']

    loss12 = criterion(pred1, proj2.detach())
    loss21 = criterion(pred2, proj1.detach())
    loss13 = criterion(pred1, proj3.detach())
    loss31 = criterion(pred3, proj1.detach())
    loss23 = criterion(pred2, proj3.detach())
    loss32 = criterion(pred3, proj2.detach())

    return loss12, loss21, loss13, loss31, loss23, loss32

def maximize_cosine_similarity(outputs1, outputs2, outputs3):
    loss12, loss21, loss13, loss31, loss23, loss32 = cosine_similarity(outputs1, outputs2, outputs3)

    loss = - (loss12.mean() + loss21.mean() +
              loss13.mean() + loss31.mean() +
              loss23.mean() + loss32.mean()) / 6.

    features = {'p_clean': F.softmax(outputs1['prediction'], dim=1),
                'p_aug1': F.softmax(outputs2['prediction'], dim=1),
                'p_aug2': F.softmax(outputs3['prediction'], dim=1),
                }

    return loss, features

def maximize_cosine_similarity_abs(outputs1, outputs2, outputs3):
    loss12, loss21, loss13, loss31, loss23, loss32 = cosine_similarity(outputs1, outputs2, outputs3)

    loss = - (loss12.abs().mean() + loss21.abs().mean() +
              loss13.abs().mean() + loss31.abs().mean() +
              loss23.abs().mean() + loss32.abs().mean()) / 6.

    features = {'p_clean': F.softmax(outputs1['prediction'], dim=1),
                'p_aug1': F.softmax(outputs2['prediction'], dim=1),
                'p_aug2': F.softmax(outputs3['prediction'], dim=1),
                }

    return loss, features

def minimize_cosine_similarity(outputs1, outputs2, outputs3):
    loss12, loss21, loss13, loss31, loss23, loss32 = cosine_similarity(outputs1, outputs2, outputs3)

    loss = (loss12.mean() + loss21.mean() +
            loss13.mean() + loss31.mean() +
            loss23.mean() + loss32.mean()) / 6.

    features = {'p_clean': F.softmax(outputs1['prediction'], dim=1),
                'p_aug1': F.softmax(outputs2['prediction'], dim=1),
                'p_aug2': F.softmax(outputs3['prediction'], dim=1),
                }

    return loss, features

def minimize_cosine_similarity_abs(outputs1, outputs2, outputs3):
    loss12, loss21, loss13, loss31, loss23, loss32 = cosine_similarity(outputs1, outputs2, outputs3)

    loss = (loss12.abs().mean() + loss21.abs().mean() +
            loss13.abs().mean() + loss31.abs().mean() +
            loss23.abs().mean() + loss32.abs().mean()) / 6.

    features = {'p_clean': F.softmax(outputs1['prediction'], dim=1),
                'p_aug1': F.softmax(outputs2['prediction'], dim=1),
                'p_aug2': F.softmax(outputs3['prediction'], dim=1),
                }

    return loss, features
