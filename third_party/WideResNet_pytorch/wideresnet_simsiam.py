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


class WideResNetSimsiam(nn.Module):
    """WideResNet simsiam class."""

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, dim=10):
        super(WideResNetSimsiam, self).__init__()
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
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        # self.fc1 = nn.Linear(n_channels[3], 2)
        # self.fc2 = nn.Linear(2, num_classes)

        prev_dim = n_channels[3]
        self.fc1 = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                 nn.BatchNorm1d(prev_dim),
                                 nn.ReLU(inplace=True),  # first layer
                                 nn.Linear(prev_dim, prev_dim, bias=False),
                                 nn.BatchNorm1d(prev_dim),
                                 nn.ReLU(inplace=True),  # second layer
                                 nn.Linear(prev_dim, dim, bias=False),
                                 nn.BatchNorm1d(dim, affine=False))  # output layer
        self.fc2 = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                 nn.BatchNorm1d(dim),
                                 nn.ReLU(inplace=True), # hidden layer
                                 nn.Linear(dim, num_classes)) # ouput layer

        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.wandb_input = dict()

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)

        return out

    def forward(self, x, targets=None):
        x_orig, x_aug1, x_aug2 = torch.chunk(x, 3)
        f_orig = self.extract_features(x_orig)
        f_aug1 = self.extract_features(x_aug1)
        f_aug2 = self.extract_features(x_aug2)

        z_orig = self.fc1(f_orig)
        z_aug1 = self.fc1(f_aug1)
        z_aug2 = self.fc1(f_aug2)

        p_orig = self.fc2(z_orig)
        p_aug1 = self.fc2(z_aug1)
        p_aug2 = self.fc2(z_aug2)

        orig = (p_orig, z_orig)
        aug1 = (p_aug1, z_aug1)
        aug2 = (p_aug2, z_aug2)

        return orig, aug1, aug2


