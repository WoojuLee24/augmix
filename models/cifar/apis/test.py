import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from datasets.concatdataset import ConcatDataset
from losses import get_additional_loss, CenterLoss


def test(net, test_loader, args):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    wandb_features = dict()
    confusion_matrix = torch.zeros(10, 10)
    tsne_features = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
            # plt = plot_tsne(net.module.features, targets)
            # plt.savefig("/ws/data/log/debug.jpg")
            # tsne_features.append(net.module.features)
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    wandb_features['test/loss'] = total_loss / len(test_loader.dataset)
    wandb_features['test/error'] = 100 - 100. * total_correct / len(test_loader.dataset)
    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset), wandb_features, confusion_matrix


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def test_c(net, test_data, args, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    wandb_features = dict()
    wandb_plts = dict()
    wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
    confusion_matrices = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc, _, confusion_matrix = test(net, test_loader,args)
        # wandb_features['test_c/{}.loss'.format(corruption)] = test_loss
        # wandb_features['test_c/{}.error'.format(corruption)] = 100 - 100. * test_acc
        wandb_table[corruption]['loss'] = test_loss
        wandb_table[corruption]['error'] = 100 - 100. * test_acc
        # wandb_plts[corruption] = confusion_matrix
        corruption_accs.append(test_acc)
        confusion_matrices.append(confusion_matrix.cpu().detach().numpy())
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    # return np.mean(corruption_accs), wandb_features
    return np.mean(corruption_accs), wandb_table, np.mean(confusion_matrices, axis=0)


def test_c_dg(net, test_data, args, corr1_data, corr2_data, base_path):
    """
    Evaluate additional loss on given combinations of corrupted datasets.
    Each corrupted dataset are compared with the same level corrupted dataset.
    """
    wandb_features = dict()
    total_additional_loss = 0.
    from itertools import combinations
    wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=CORRUPTIONS)
    for corruption1, corruption2 in combinations(CORRUPTIONS, 2):
        clean_loss = 0.
        corr_additional_loss = 0.
        test_data.data = np.load(base_path + 'clean.npy')
        corr1_data.data = np.load(base_path + corruption1 + '.npy')
        corr2_data.data = np.load(base_path + corruption2 + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        corr1_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        corr2_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        concat_data = ConcatDataset((test_data, corr1_data, corr2_data))

        test_loader = torch.utils.data.DataLoader(
            concat_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)
        # test_loss, test_acc, _ = test(net, test_loader)
        with torch.no_grad():
            for clean, corr1, corr2 in test_loader:
                images = torch.cat([clean[0], corr1[0], corr2[0]], dim=0)
                targets = torch.cat([clean[1], corr1[1], corr2[1]], dim=0)
                images, targets = images.cuda(), targets.cuda()
                logits = net(images)
                logits_clean, logits_aug1, logits_aug2 = torch.chunk(logits, 3)
                target_clean, target_aug1, target_aug2 = torch.chunk(targets, 3)
                loss = F.cross_entropy(logits_clean, target_clean)
                additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2)
                clean_loss += float(loss.data)
                corr_additional_loss += float(additional_loss.data)
            # wandb_features['test_c/loss_clean'] = clean_loss / len(test_loader)
            # wandb_features['test_c/additional_loss_{}_{}'.format(corruption1, corruption2)] = \
            #     corr_additional_loss / len(test_loader)
            wandb_table[corruption1][corruption2] = corr_additional_loss / len(test_loader)
            print('test_c/loss_clean: ', clean_loss / len(test_loader))
            print('test_c/additional_loss_{}_{}'.format(corruption1, corruption2), corr_additional_loss / len(test_loader))

        total_additional_loss += corr_additional_loss
    combinations_length = sum(1 for _ in combinations(CORRUPTIONS, 2))
    # wandb_features['test_c/additional_loss_total'.format(total_additional_loss)] = \
    #     total_additional_loss / combinations_length
    print('test_c/additional_loss_total'.format(total_additional_loss), total_additional_loss / combinations_length)

    # wandb_table = wandb.Table(data=df)

    return total_additional_loss / combinations_length, wandb_features, wandb_table
