import torch
from torch import nn

class FeatureHook(nn.Module):

    def __init__(self, layer_list):
        self.layer_list = layer_list

    def hook_layer(self, model, selected_layer):
        def hook_function(module, grad_in, grad_out):
            # Gets output of the selected layer
            if not selected_layer in model.hook_features:
                model.hook_features[selected_layer] = []

            model.hook_features[selected_layer].append(grad_out)

        # Hook the selected layer
        for n, m in model.named_modules():
            if n == str(selected_layer):
                m.register_forward_hook(hook_function)

    def hook_multi_layer(self, model):
        for layer_name in self.layer_list:
            self.hook_layer(model, layer_name)