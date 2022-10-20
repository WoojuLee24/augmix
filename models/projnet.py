import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(t):
    return t.reshape(t.shape[0], -1)

class NetWrapper(nn.Module):
    def __init__(self, net, layer):
        super(NetWrapper, self).__init__()
        self.net = net
        self.layer = layer

        self.hidden = dict()
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        return representation

class _projNet(nn.Module):
    def __init__(self,
                 base_encoder,
                 projector,
                 predictor,
                 hidden_layer=-2,
                 criterion=None):
        super(_projNet, self).__init__()
        self.encoder = NetWrapper(base_encoder, layer=hidden_layer)
        self.projector = projector
        self.predictor = predictor

        self.outputs = dict()

    def forward(self, x):
        self.outputs.clear()

        representation = self.encoder(x)
        # E.g., representation.shape = (512*3, 128, 8, 8)
        projection = self.projector(representation.reshape(representation.shape[0], -1))
        prediction = self.predictor(projection)

        self.outputs.update({'representation': representation,
                             'projection': projection,
                             'prediction': prediction}) # Acts like a hook
        return self.outputs


def print_layer_name(net):
    modules = dict([*net.named_modules()])
    for key, value in modules.items():
        print(key)


def print_layer_index(net):
    children = [*net.children()]
    for i in range(len(children)):
        print(f"[{i}]---------------")
        print(children[i])


def get_dim(net, layer, flattened=True):
    if type(layer) == str:
        modules = dict([*net.named_modules()])
        weight = modules.get(layer, None).weight
    elif type(layer) == int:
        children = [*net.children()]
        weight = children[layer].weight
    else:
        raise TypeError(f'The type of layer must be str or int,'
                        f'but got {type(layer)}')

    if flattened:
        return flatten(weight).shape
    else:
        return weight.shape


'''
* Please use this projNet with version like below...
* How to find hidden layer name (or index)?
    please use `print_layer_name(net)` or `print_layer_index(net)` 
    on your terminal
'''
from third_party.WideResNet_pytorch.wideresnet import WideResNet
def projNetv1(args,
              pred_dim,
              # hidden_dim=2048,
              ):
    ''' WideResNet:
        ...
        block3.layer.5.relu2
        block3.layer.5.conv2
        bn1
        relu
        avgpool
        fc
    '''
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_2(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_3(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_4(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_5(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


class _simple_resblock(nn.Module):
    def __init__(self, middle_layers, last_layers):
        super(_simple_resblock, self).__init__()
        self.middle_layers = middle_layers
        self.last_layers = last_layers

    def forward(self, x):
        identity = x

        x = self.middle_layers(x)

        out = x + identity

        if self.last_layers is not None:
            out = self.last_layers(out)

        return out


def projNetv1_6(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     )
    projector_last_layers = nn.Sequential(encoder.fc,
                                          nn.BatchNorm1d(args.hidden_dim, affine=False))
    projector = _simple_resblock(projector_layers, projector_last_layers)

    predictor_layers = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                                     nn.BatchNorm1d(pred_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_6_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     )
    projector_last_layers = nn.Sequential(encoder.fc,
                                          nn.BatchNorm1d(args.hidden_dim, affine=False))
    projector = _simple_resblock(projector_layers, projector_last_layers)

    predictor = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_6_2(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False)
                              )

    predictor_layers = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                                     nn.BatchNorm1d(pred_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_1_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              )
    projector_last_layers = nn.Sequential(encoder.fc,
                                          nn.BatchNorm1d(args.hidden_dim, affine=False),)
    projector = _simple_resblock(projector_layers, projector_last_layers)

    predictor_layers = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_1_2(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     )
    projector_last_layers = nn.Sequential(encoder.fc,
                                          nn.BatchNorm1d(args.hidden_dim, affine=False),)
    projector = _simple_resblock(projector_layers, projector_last_layers)

    predictor = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                              nn.BatchNorm1d(pred_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_1_3(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                              nn.BatchNorm1d(prev_dim),
                              nn.ReLU(inplace=True),
                              encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False)
                              )

    predictor_layers = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                                     nn.BatchNorm1d(pred_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_2_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor_layers = nn.Sequential(nn.Linear(args.hidden_dim, pred_dim, bias=False),
                                     nn.BatchNorm1d(pred_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_3_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector = nn.Sequential(encoder.fc,
                              nn.BatchNorm1d(args.hidden_dim, affine=False),
                              )

    predictor_layers = nn.Sequential(nn.Linear(pred_dim, pred_dim))
    predictor = _simple_resblock(predictor_layers, last_layers=None)

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_4_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     )
    projector_last_layers = nn.Sequential(encoder.fc,
                                          nn.BatchNorm1d(args.hidden_dim, affine=False),)
    projector = _simple_resblock(projector_layers, last_layers=projector_last_layers)

    predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)


def projNetv1_5_1(args, pred_dim,):
    encoder = WideResNet(depth=args.layers,
                         num_classes=args.hidden_dim,
                         widen_factor=args.widen_factor,
                         drop_rate=args.droprate)
    prev_dim = get_dim(encoder, 'fc', flattened=True)[-1]

    projector_layers = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(prev_dim, prev_dim, bias=False),
                                     nn.BatchNorm1d(prev_dim),
                                     nn.ReLU(inplace=True),
                                     )
    projector_last_layers = nn.Sequential(encoder.fc,nn.BatchNorm1d(args.hidden_dim, affine=False),)
    projector = _simple_resblock(projector_layers, last_layers=projector_last_layers)

    predictor = nn.Sequential(nn.Linear(pred_dim, pred_dim))

    return _projNet(encoder, projector, predictor, hidden_layer='avgpool',)

