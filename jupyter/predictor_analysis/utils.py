import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def cls_id_to_name(id):
    return class_name[id]


def visualize(images, num_rows=1, num_cols=1, figsize_unit=(7, 7),
              imshow=True,
              title='', xlabels=None, ylabels=None):
    figsize = (figsize_unit[0] * num_rows, figsize_unit[1] * num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            image = images[idx]
            # Set x_label
            if xlabels is not None:
                xlabel = xlabels[idx]
            else:
                xlabel = ''
            if ylabels is not None:
                ylabel = ylabels[idx]
            else:
                ylabel = ''

            # Valid shape: (H, W, 3)
            if torch.is_tensor(image):
                image = image.permute(1, 2, 0)

            # Valid type: np.array
            if images[idx].device.type == 'cuda':
                image = image.cpu().detach().numpy()

            # Image show
            if (num_rows == 1) and (num_cols == 1):
                axes.imshow(image)
                axes.xaxis.set_ticks([])
                axes.yaxis.set_ticks([])
                axes.set_xlabel(xlabel)
                axes.set_ylabel(ylabel)
            elif num_rows == 1:
                axes[col].imshow(image)
                axes[col].xaxis.set_ticks([])
                axes[col].yaxis.set_ticks([])
                axes[col].set_xlabel(xlabel, fontsize=10)
                axes[col].set_ylabel(ylabel, fontsize=10)
            elif num_cols == 1:
                axes[row].imshow(image)
                axes[row].xaxis.set_ticks([])
                axes[row].yaxis.set_ticks([])
                axes[row].set_xlabel(xlabel)
                axes[row].set_ylabel(ylabel)
            else:
                axes[row, col].imshow(image)
                axes[row, col].xaxis.set_ticks([])
                axes[row, col].yaxis.set_ticks([])
                axes[row, col].set_xlabel(xlabel)
                axes[row, col].set_ylabel(ylabel)

            idx += 1
    if imshow:
        fig.show()
    else:
        return fig, axes


def get_images(original_images, hard_examples, device='cuda'):
    if torch.is_tensor(original_images):
        x_clean = original_images.to(device)  # (B, C, H, W)
        images = [x_clean]
    elif isinstance(original_images, tuple):
        x_clean = original_images[0].to(device)  # (B, C, H, W)
        x_aug1 = original_images[1].to(device)
        x_aug2 = original_images[2].to(device)
        images = [x_clean, x_aug1, x_aug2]
    else:
        raise TypeError('only supported for tensor or tuple')

    hard_images = []
    for i in range(len(hard_examples)):
        hard_examples[i]['image'] = hard_examples[i]['image'].to(device)
        hard_images.append(hard_examples[i]['image'])

    images.extend(hard_images)
    return images


def get_types(original_images, hard_examples):
    if torch.is_tensor(original_images):
        types = ['clean']
    elif isinstance(original_images, tuple):
        types = ['clean', 'aug1', 'aug2']
    else:
        raise TypeError('only supported for tensor or tuple')

    hard_types = []
    for i in range(len(hard_examples)):
        hard_types.append(f"{hard_examples[i]['corruption']}{hard_examples[i]['severity']}")

    types.extend(hard_types)
    return types


def get_targets(original_images, original_targets, hard_examples):
    original_type = cls_id_to_name(original_targets)
    if torch.is_tensor(original_images):
        targets = [original_type]
    elif isinstance(original_images, tuple):
        targets = [original_type, original_type, original_type]
    else:
        raise TypeError('only supported for tensor or tuple')

    hard_targets = []
    for i in range(len(hard_examples)):
        hard_target = cls_id_to_name(hard_examples[i]['pred'])
        hard_targets.append(hard_target)

    targets.extend(hard_targets)
    return targets


from matplotlib import cm
def visualize_module_output(hook_results, layer_name, xticks=[], xlabels=None, ylabels=None, log=False, imshow=True):
    shape = hook_results[0]['hook_result'][layer_name].shape
    if len(shape) > 2:
        b, c, h, w = shape
        if log:
            print(f"{layer_name} ({b},{c},{h},{w})")
        if h == 1 and w == 1:
            dim = 1
        else:
            dim = 2
    else:
        b, c = shape
        dim = 1
        if log:
            print(f"{layer_name} ({b},{c})")

    outputs = []
    for i in range(len(hook_results)):
        output = hook_results[i]['hook_result'][layer_name]
        if dim == 1:
            output = output.view(b, -1)
            output = output.repeat(4, 1)
            outputs.append(output)
            if log:
                print(f"output.shape = {output.shape}")
        elif dim == 2:
            output = output.view(b, c, -1)
            output = torch.mean(output, dim=-1, keepdim=True).squeeze(-1)
            # outputs.append(torch.sum(output[0], dim=0, keepdim=True).squeeze(0))
            output = output.repeat(4, 1)
            outputs.append(output)

    _len = len(outputs)
    fig, axes = plt.subplots(_len, 1, figsize=(_len * 3, 2))
    for i in range(_len):
        axes[i].imshow(outputs[i].cpu().detach().numpy(), cmap=cm.Blues)
        axes[i].xaxis.set_ticks(xticks)
        axes[i].yaxis.set_ticks([])
        if xlabels is not None:
            axes[i].set_xlabel(xlabels[i])
        if ylabels is not None:
            axes[i].set_ylabel(ylabels[i])

    fig.suptitle(layer_name)
    if imshow:
        fig.show()
    return fig,


import torchvision
def denormalize(images):
    denormalize = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.0, 0.0, 0.0,], [1/0.5, 1/0.5, 1/0.5]),
        torchvision.transforms.Normalize([-0.5, -0.5, -0.5], [1.0, 1.0, 1.0]),])
    denormalized_images = []
    for image in images:
        denormalized_images.append(denormalize(image))
    return denormalized_images


import copy
def get_data(dataset, index, config, imshow=False):
    ### Dataset
    original_images, original_targets, hard_examples = dataset[index]
    images = get_images(original_images, hard_examples, config.device)
    types = get_types(original_images, hard_examples)
    targets = get_targets(original_images, original_targets, hard_examples)

    if imshow:
        visualize(denormalize(images), 1, len(images), xlabels=types, ylabels=targets)
    return images, types, targets

def inference(images, types, model, param_manager):
    ### Model
    hook_results, logits, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(images)):
            logit = model(images[i].unsqueeze(0))
            prob = F.softmax(logit, dim=-1)

            logits.append(logit.cpu().detach())
            probs.append(prob.cpu().detach())

            results = {'name': types[i],
                       'logit': logit.cpu().detach(),
                       'prob': prob.cpu().detach(),
                       'hook_result': copy.deepcopy(param_manager.hook_results['output'])}
            hook_results.append(results)

    return probs