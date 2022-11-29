"""WideResNet implementation (https://arxiv.org/abs/1605.07146)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  """Basic ResNet block."""

  def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
    super(BasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.drop_rate = drop_rate
    self.is_in_equal_out = (in_planes == out_planes)
    self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False) or None

  def forward(self, x):
    if not self.is_in_equal_out:
      x = self.relu1(self.bn1(x))
    else:
      out = self.relu1(self.bn1(x))
    if self.is_in_equal_out:
      out = self.relu2(self.bn2(self.conv1(out)))
    else:
      out = self.relu2(self.bn2(self.conv1(x)))
    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)
    out = self.conv2(out)
    if not self.is_in_equal_out:
      return torch.add(self.conv_shortcut(x), out)
    else:
      return torch.add(x, out)


class NetworkBlock(nn.Module):
  """Layer container for blocks."""

  def __init__(self,
               nb_layers,
               in_planes,
               out_planes,
               block,
               stride,
               drop_rate=0.0):
    super(NetworkBlock, self).__init__()
    self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                  stride, drop_rate)

  def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                  drop_rate):
    layers = []
    for i in range(nb_layers):
      layers.append(
          block(i == 0 and in_planes or out_planes, out_planes,
                i == 0 and stride or 1, drop_rate))
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.layer(x)


class WideResNetFc(nn.Module):
    """WideResNet class."""

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):
        super(WideResNetFc, self).__init__()
        self.hook_features = dict()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                                   drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                                   drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                                   drop_rate)
        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(n_channels[3], num_classes)
        self.fc1 = nn.Linear(n_channels[3], n_channels[3])
        self.bn1 = nn.BatchNorm1d(n_channels[3])
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_channels[3], n_channels[3])
        self.bn2 = nn.BatchNorm1d(n_channels[3])
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(n_channels[3], n_channels[3])
        self.bn3 = nn.BatchNorm1d(n_channels[3])
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(n_channels[3], num_classes)

        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.wandb_input = dict()

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        return out

    def pooling_features(self, x):
        out = self.relu(self.bn(x))
        out = self.avgpool(out)
        out = out.view(-1, self.n_channels)

        return out

    def classifier(self, x):
        out = self.relu1(self.bn1(self.fc1(x)))
        out = self.relu2(self.bn2(self.fc2(out)))
        out = self.relu3(self.bn3(self.fc3(out)))
        out = self.fc4(out)

        return out


    def forward(self, x, fcnoise=None, fcnoise_s=None, targets=None):
        features = self.extract_features(x)

        if fcnoise == 'unoise':
            # channel noise
            B, C, H, W = features.size()
            noise = fcnoise_s * (2 * torch.rand(B, C).to('cuda') - 1)
            noise = noise.unsqueeze(dim=2).unsqueeze(dim=3)
            noise = noise.repeat((1, 1, H, W))
            noise_npy = noise.detach().cpu().numpy()
            features += noise

        pooled_features = self.pooling_features(features)
        # logits = self.fc(self.features)
        logits = self.classifier(pooled_features)




        if targets is not None:
            from utils.visualize import plot_tsne, multi_plot_tsne
            targets_all = torch.cat((targets, targets, targets), 0)

            input_list = [self.features, logits]
            targets_list = [targets_all, targets_all]
            title_list = ['features', 'logits']
            plt, fig = multi_plot_tsne(input_list, targets_list, title_list, rows=1, cols=2,
                                       # perplexity=50, n_iter=300)
                                       perplexity=30, n_iter=300)
            self.wandb_input['tsne'] = fig
            plt.close(fig)

        return logits
