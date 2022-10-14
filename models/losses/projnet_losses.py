import torch
import torch.nn as nn
import torch.nn.functional as F

class _projNetLoss(nn.Module):
    def __init__(self,
                 criterion,
                 additional_criterion,
                 input1=None,
                 input2=None,
                 input3=None,
                 lambda_weight=1e-1):
        super(_projNetLoss, self).__init__()
        self.criterion = criterion
        self.additional_criterion = additional_criterion
        self.lambda_weight = lambda_weight

        # They will be the input of additional criterion
        # They can be 'projection' or 'prediction'.
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3

        self.outputs = dict()

    def forward(self, outputs1, outputs2=None, outputs3=None, targets=None):
        self.outputs.clear()
        original_loss, additional_loss = torch.tensor(0.).cuda(), torch.tensor(0.).cuda(),
        if targets is not None:
            original_loss = self.criterion(outputs1['prediction'], targets)
        if outputs2 is not None:
            if outputs3 is None:
                additional_loss, features = self.additional_criterion(outputs1[self.input1], outputs2[self.input2])
                self.outputs.update(features)
            else:
                additional_loss, features = self.additional_criterion(outputs1[self.input1], outputs2[self.input2], outputs3[self.input3])
                self.outputs.update(features)

        loss = original_loss + self.lambda_weight * additional_loss

        self.outputs.update({'total_loss': loss,
                             'original_loss': original_loss,
                             'additional_loss': additional_loss}) # Acts like a hook

        return loss


from models.losses.jsd import jsd


def projNetLoss(name, lambda_weight=12):
    if name == 'projnetv1.pred':
        return projNetLossv1(lambda_weight)
    elif name == 'projnetv1.test':
        return projNetLossTest(lambda_weight)
    elif name == 'projnetv1.proj':
        return projNetLossv1_proj(lambda_weight)
    elif name == 'projnetv1.repr':
        return projNetLossv1_repr(lambda_weight)

def projNetLossv1(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd

    return _projNetLoss(criterion, additional_criterion,
                        input1='prediction', input2='prediction', input3='prediction',
                        lambda_weight=lambda_weight)

def projNetLossTest(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd

    return _projNetLoss(criterion, additional_criterion,
                        input1='prediction', lambda_weight=lambda_weight)


def projNetLossv1_proj(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd

    return _projNetLoss(criterion, additional_criterion,
                        input1='projection', input2='projection', input3='projection',
                        lambda_weight=lambda_weight)

def projNetLossv1_repr(lambda_weight=12):
    criterion = F.cross_entropy
    additional_criterion = jsd

    return _projNetLoss(criterion, additional_criterion,
                        input1='representation', input2='representation', input3='representation',
                        lambda_weight=lambda_weight)
