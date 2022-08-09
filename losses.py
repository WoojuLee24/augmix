import torch
import torch.nn.functional as F
import torch.nn as nn


def get_additional_loss(name, logits_clean, logits_aug1, logits_aug2, lambda_weight=12, targets=None, temper=1, **kwargs):

    if name == 'none':
        loss = 0
    elif name == 'jsd':
        loss = jsd(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'jsd_temper':
        loss = jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight, temper)
    elif name == 'kl':
        loss = kl(logits_clean, logits_aug1, logits_aug2, lambda_weight)
    elif name == 'ntxent':
        loss = ntxent(logits_clean, logits_aug1, logits_aug2, lambda_weight, targets)

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

def jsd_distance(logit1, logit2, reduction='batchmean'):
    p1, p2 = F.softmax(logit1, dim=1), F.softmax(logit2, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p1+ p2) / 2., 1e-7, 1).log()
    loss =  (F.kl_div(p_mixture, p1, reduction=reduction) + F.kl_div(p_mixture, p2, reduction=reduction)) / 2.

    return loss

def jsd_temper(logits_clean, logits_aug1, logits_aug2, lambda_weight=12, temper=0.5):
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

