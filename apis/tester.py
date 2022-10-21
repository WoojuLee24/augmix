import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from utils.utils import CORRUPTIONS
from datasets.concatdataset import ConcatDataset
from losses import get_additional_loss
import math
from collections import defaultdict

class Tester():
    def __init__(self, net, args, wandb_logger=None, device='cuda'):
        self.net = net
        self.args = args
        self.wandb_logger = wandb_logger
        self.device = device

    def __call__(self, data_loader):
        pass

    def test_c(self, test_dataset, base_path=None):
        """Evaluate network on given corrupted dataset."""
        wandb_features, wandb_plts = dict(), dict()
        wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
        confusion_matrices = []
        if self.args.dataset == 'cifar10' or 'cifar100':
            corruption_accs = []
            for corruption in CORRUPTIONS:
                # Reference to original data is mutated
                test_dataset.data = np.load(base_path + corruption + '.npy')
                test_dataset.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True)

                test_loss, test_acc, _, confusion_matrix = self.test(test_loader)

                wandb_table[corruption]['loss'] = test_loss
                wandb_table[corruption]['error'] = 100 - 100. * test_acc
                # wandb_plts[corruption] = confusion_matrix

                corruption_accs.append(test_acc)
                confusion_matrices.append(confusion_matrix.cpu().detach().numpy())
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    corruption, test_loss, 100 - 100. * test_acc))

            # return np.mean(corruption_accs), wandb_features
            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm
        else:  # imagenet
            corruption_accs = {}
            for c in CORRUPTIONS:
                print(c)
                for s in range(1, 6):
                    valdir = os.path.join(self.args.corrupted_data, c, str(s))
                    val_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.args.eval_batch_size,
                        shuffle=False,
                        num_workers=self.args.num_workers,
                        pin_memory=True)

                    loss, acc1, _, _ = self.test(val_loader)
                    if c in corruption_accs:
                        corruption_accs[c].append(acc1)
                    else:
                        corruption_accs[c] = [acc1]

                    print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
                        s, loss, 100. * acc1))
            return corruption_accs

    def test(self, data_loader, data_type='clean'):
        """Evaluate network on given dataset."""
        self.net.eval()
        total_loss, total_correct = 0., 0.
        wandb_features = dict()
        confusion_matrix = torch.zeros(10, 10)
        tsne_features = []
        with torch.no_grad():
            for images, targets in data_loader:
                images, targets = images.cuda(), targets.cuda()
                logits = self.net(images)

                if self.args.analysis:
                    from utils.visualize import multi_plot_tsne
                    input_list = [self.net.module.features, logits]
                    targets_list = [targets, targets]
                    title_list = ['features', 'logits']
                    save_path = os.path.join(self.args.save, 'analysis', data_type + '.jpg')
                    tsne, fig = multi_plot_tsne(input_list, targets_list, title_list, rows=1, cols=2,
                                                perplexity=30, n_iter=300,
                                                save=save_path, log_wandb=self.args.wandb, data_type=data_type)

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        datasize = len(data_loader.dataset)
        wandb_features['test/loss'] = total_loss / datasize
        wandb_features['test/error'] = 100 - 100. * total_correct / datasize
        test_loss = total_loss / datasize
        test_acc = total_correct / datasize
        return test_loss, test_acc, wandb_features, confusion_matrix


    def test_c_v2(self, test_dataset, base_path=None):
        """
        Evaluate network on given corrupted dataset.
        Evaluate the jsd distance between original data and corrupted data.
        """
        test_c_features, test_c_mean_features = dict(), defaultdict(float)
        wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
        confusion_matrices = []
        import copy
        original_test_dataset = copy.deepcopy(test_dataset)
        original_test_dataset.data = np.load(base_path + 'clean' + '.npy')
        original_test_dataset.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        k = len(CORRUPTIONS)
        if self.args.dataset == 'cifar10' or 'cifar100':
            corruption_accs = []
            for corruption in CORRUPTIONS:
                # Reference to original data is mutated
                test_dataset.data = np.load(base_path + corruption + '.npy')
                test_dataset.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
                concat_dataset = ConcatDataset((original_test_dataset, test_dataset))
                test_loader = torch.utils.data.DataLoader(
                    concat_dataset,
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True)

                test_loss, test_acc, test_c_feature, confusion_matrix = self.test_v2(test_loader, corruption)

                wandb_table[corruption]['loss'] = test_loss
                wandb_table[corruption]['error'] = 100 - 100. * test_acc
                # wandb_plts[corruption] = confusion_matrix

                for key, value in test_c_feature.items():
                    corruption_key = f'test/{corruption}/{key}'
                    test_c_features[corruption_key] = value
                for key, value in test_c_feature.items():
                    key = f'test/mean_corruption/{key}'
                    test_c_mean_features[key] += value

                corruption_accs.append(test_acc)
                confusion_matrices.append(confusion_matrix.cpu().detach().numpy())
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    corruption, test_loss, 100 - 100. * test_acc))

            for key, value in test_c_mean_features.items():
                test_c_features[key] = value / len(CORRUPTIONS)

            # return np.mean(corruption_accs), wandb_features
            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm, test_c_features
        else:  # imagenet
            corruption_accs = {}
            for c in CORRUPTIONS:
                print(c)
                for s in range(1, 6):
                    valdir = os.path.join(self.args.corrupted_data, c, str(s))
                    val_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.args.eval_batch_size,
                        shuffle=False,
                        num_workers=self.args.num_workers,
                        pin_memory=True)

                    loss, acc1, _, _ = self.test(val_loader)
                    if c in corruption_accs:
                        corruption_accs[c].append(acc1)
                    else:
                        corruption_accs[c] = [acc1]

                    print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
                        s, loss, 100. * acc1))
            return corruption_accs


    def test_v2(self, data_loader, data_type='clean'):
        """Evaluate network on given concatenated dataset."""
        self.net.eval()
        total_loss, total_correct = 0., 0.
        wandb_features = dict()
        confusion_matrix = torch.zeros(10, 10)
        tsne_features = []
        with torch.no_grad():
            for i, (clean_data, corrupted_data) in enumerate(data_loader):
                clean_images, _ = clean_data
                corrupted_images, targets = corrupted_data

                images_all = torch.cat([clean_images, corrupted_images], 0).to(self.device)
                # targets_all = torch.cat([targets, targets], 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all)
                logits, logits_corrupted = torch.chunk(logits_all, 2)

                additional_loss, feature = get_additional_loss(self.args,
                                                               logits, logits_corrupted, None,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)

                # if self.args.analysis:
                #     from utils.visualize import multi_plot_tsne
                #     input_list = [self.net.module.features, logits]
                #     targets_list = [targets, targets]
                #     title_list = ['features', 'logits']
                #     save_path = os.path.join(self.args.save, 'analysis', data_type + '.jpg')
                #     tsne, fig = multi_plot_tsne(input_list, targets_list, title_list, rows=1, cols=2,
                #                                 perplexity=30, n_iter=300,
                #                                 save=save_path, log_wandb=self.args.wandb, data_type=data_type)

                loss = F.cross_entropy(logits_corrupted, targets)
                pred = logits_corrupted.data.max(1)[1]
                total_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

                if i == 0:
                    for key, value in feature.items():
                        # total_key = f'test/{data_type}_total_' + key
                        # wandb_features[total_key] = feature[key].detach()
                        wandb_features[key] = feature[key].detach()
                else:
                    # exclude terminal data for wandb_features: batch size is different.
                    if logits_corrupted.size(0) == self.args.eval_batch_size:
                        for key, value in feature.items():
                            # total_key = f'test/{data_type}_total_' + key
                            # wandb_features[total_key] += feature[key].detach()
                            wandb_features[key] += feature[key].detach()

                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # logging total results
        # features
        denom = math.floor(len(data_loader.dataset) / self.args.eval_batch_size)
        for key, value in wandb_features.items():
            wandb_features[key] = wandb_features[key] / denom
        wandb_features['p_clean_sample'] = feature['p_clean']
        wandb_features['p_aug1_sample'] = feature['p_aug1']

        # loss
        denom = len(data_loader.dataset) / self.args.eval_batch_size
        test_loss = total_loss / denom

        # error
        datasize = len(data_loader.dataset)
        test_acc = total_correct / datasize

        return test_loss, test_acc, wandb_features, confusion_matrix

