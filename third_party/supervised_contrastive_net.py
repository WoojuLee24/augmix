import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, model_fun, head='mlp', in_feature=2048, out_feature=128):
        super(SupConNet, self).__init__()
        self.encoder = model_fun
        if head == 'linear':
            self.head = nn.Linear(in_feature, out_feature)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_feature, in_feature),
                nn.ReLU(inplace=True),
                nn.Linear(in_feature, out_feature)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
