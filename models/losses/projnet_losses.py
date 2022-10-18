import torch
import torch.nn as nn
import torch.nn.functional as F

class _projNetLoss(nn.Module):
    def __init__(self,
                 criterion,
                 additional_criterion,
                 lambda_weight=1e-1):
        super(_projNetLoss, self).__init__()
        self.criterion = criterion
        self.additional_criterion = additional_criterion
        self.lambda_weight = lambda_weight

        self.outputs = dict()

    def forward(self, outputs1, outputs2=None, outputs3=None, targets=None):
        self.outputs.clear()
        original_loss, additional_loss = torch.tensor(0.).cuda(), torch.tensor(0.).cuda(),
        if targets is not None:
            original_loss = self.criterion(outputs1['prediction'], targets)
        if outputs2 is not None:
            if outputs3 is None:
                additional_loss, features = self.additional_criterion(outputs1, outputs2)
                self.outputs.update(features)
            else:
                additional_loss, features = self.additional_criterion(outputs1, outputs2, outputs3)
                self.outputs.update(features)

        loss = original_loss + self.lambda_weight * additional_loss

        self.outputs.update({'total_loss': loss,
                             'original_loss': original_loss,
                             'additional_loss': self.lambda_weight * additional_loss}) # Acts like a hook

        return loss


def projNetLoss(name, lambda_weight=12):
    # Version1
    if name == 'projnetv1.pred':
        return projNetLossv1(lambda_weight)
    elif name == 'projnetv1.test':
        return projNetLossTest(lambda_weight)
    elif name == 'projnetv1.proj':
        return projNetLossv1_proj(lambda_weight)
    elif name == 'projnetv1.repr':
        return projNetLossv1_repr(lambda_weight)
    # Version 1.1~
    elif name == 'projnetv1.1':
        return projNetLossv1_1(lambda_weight)
    elif name == 'projnetv1.2':
        return projNetLossv1_2(lambda_weight)


# Version 1
from models.losses.losses import jsd_pred
def projNetLossv1(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd_pred

    return _projNetLoss(criterion, additional_criterion,
                        lambda_weight=lambda_weight)

def projNetLossTest(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd_pred

    return _projNetLoss(criterion, additional_criterion, lambda_weight=lambda_weight)

from models.losses.losses import jsd_proj
def projNetLossv1_proj(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd_proj

    return _projNetLoss(criterion, additional_criterion,
                        lambda_weight=lambda_weight)

from models.losses.losses import jsd_repr
def projNetLossv1_repr(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd_repr

    return _projNetLoss(criterion, additional_criterion,
                        lambda_weight=lambda_weight)

# Version 1.1~
from models.losses.losses import maximize_cosine_similarity
def projNetLossv1_1(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = maximize_cosine_similarity

    return _projNetLoss(criterion, additional_criterion,
                        lambda_weight=lambda_weight)

from models.losses.losses import maximize_cosine_similarity_abs
def projNetLossv1_2(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = maximize_cosine_similarity_abs

    return _projNetLoss(criterion, additional_criterion,
                        lambda_weight=lambda_weight)

