import os
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np
from utils.utils import CORRUPTIONS
from datasets.concatdataset import ConcatDataset
from losses import get_additional_loss
import math
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, net, args, wandb_logger=None, device='cuda'):
        self.net = net
        self.args = args
        self.wandb_logger = wandb_logger
        self.device = device

        if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
            self.classes = 10
        elif args.dataset == 'imagenet':
            self.classes = 1000

    def __call__(self, data_loader):
        pass

    def test_c(self, test_dataset, base_path=None):
        """Evaluate network on given corrupted dataset."""
        wandb_features, wandb_plts = dict(), dict()
        wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
        confusion_matrices = []
        if (self.args.dataset == 'cifar10') or (self.args.dataset == 'cifar100'):
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
                confusion_matrices.append(confusion_matrix)
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    corruption, test_loss, 100 - 100. * test_acc))

            # return np.mean(corruption_accs), wandb_features
            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm
        else:  # imagenet
            corruption_accs = []

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            preprocess = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess,
            ])

            for c in CORRUPTIONS:
                print(c)
                severity_accs = []
                severity_losses = []
                for s in range(1, 6):
                    # valdir = os.path.join(self.args.corrupted_data, c, str(s))
                    valdir = os.path.join(base_path, c, str(s))
                    test_dataset = datasets.ImageFolder(valdir, test_transform)
                    val_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.args.eval_batch_size,
                        shuffle=False,
                        num_workers=self.args.num_workers,
                        pin_memory=True)

                    loss, acc1, _, confusion_matrix = self.test(val_loader)
                    severity_accs.append(acc1)
                    severity_losses.append(loss)

                test_loss = np.mean(severity_losses)
                test_acc = np.mean(severity_accs)
                wandb_table[c]['loss'] = test_loss
                wandb_table[c]['error'] = 100 - 100. * test_acc

                corruption_accs.append(test_acc)
                confusion_matrices.append(confusion_matrix)
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    c, test_loss, 100 - 100. * test_acc))

            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm


    def test(self, data_loader, data_type='clean'):
        """Evaluate network on given dataset."""
        self.net.eval()
        total_loss, total_correct = 0., 0.
        wandb_features = dict()
        confusion_matrix = torch.zeros(self.classes, self.classes)
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

                if self.args.debug == True:
                    print("debug test epoch is terminated")
                    break

        datasize = len(data_loader.dataset)
        wandb_features['test/loss'] = total_loss / datasize
        wandb_features['test/error'] = 100 - 100. * total_correct / datasize
        test_loss = total_loss / datasize
        test_acc = total_correct / datasize
        confusion_matrix = confusion_matrix.detach().cpu().numpy()
        return test_loss, test_acc, wandb_features, confusion_matrix


    def test_c_v2(self, test_dataset, base_path=None):
        """
        Evaluate network on given corrupted dataset.
        Evaluate the jsd distance between original data and corrupted data.
        """
        test_c_features, test_c_mean_features = dict(), defaultdict(float)
        wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
        confusion_matrices_mean = []
        confusion_matrices = dict()
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
                confusion_matrices_mean.append(confusion_matrix)
                confusion_matrices[f"test/{corruption}"] = confusion_matrix
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    corruption, test_loss, 100 - 100. * test_acc))

            for key, value in test_c_mean_features.items():
                test_c_features[key] = value / len(CORRUPTIONS)

            # return np.mean(corruption_accs), wandb_features
            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices_mean, axis=0)
            confusion_matrices['test/corruption_mean'] = test_c_cm
            test_c_cms = confusion_matrices
            return test_c_acc, wandb_table, test_c_cms, test_c_features
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
        confusion_matrix = torch.zeros(self.classes, self.classes)
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


    def test_v2_trainer(self, data_loader, data_type='clean'):
        """Evaluate network on given concatenated dataset."""
        self.net.eval()
        total_loss, total_correct = 0., 0.
        wandb_features = dict()
        confusion_matrices = []
        confusion_matrix = torch.zeros(self.classes, self.classes)
        confusion_matrix_aug1 = torch.zeros(self.classes, self.classes)
        confusion_matrix_aug2 = torch.zeros(self.classes, self.classes)
        confusion_matrix_pred_aug1 = torch.zeros(self.classes, self.classes)
        confusion_matrix_aug1_aug2 = torch.zeros(self.classes, self.classes)
        confusion_matrix_pred_aug2 = torch.zeros(self.classes, self.classes)


        tsne_features = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug2.data.max(1)[1]

                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)


                loss = F.cross_entropy(logits_clean, targets)
                total_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

                if i == 0:
                    for key, value in feature.items():
                        # total_key = f'test/{data_type}_total_' + key
                        # wandb_features[total_key] = feature[key].detach()
                        wandb_features[key] = feature[key].detach()
                else:
                    # exclude terminal data for wandb_features: batch size is different.
                    if logits_clean.size(0) == self.args.batch_size:
                        for key, value in feature.items():
                            # total_key = f'test/{data_type}_total_' + key
                            # wandb_features[total_key] += feature[key].detach()
                            wandb_features[key] += feature[key].detach()

                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(targets.view(-1), pred_aug1.view(-1)):
                    confusion_matrix_aug1[t.long(), p.long()] += 1
                for t, p in zip(targets.view(-1), pred_aug2.view(-1)):
                    confusion_matrix_aug2[t.long(), p.long()] += 1
                for t, p in zip(pred.view(-1), pred_aug1.view(-1)):
                    confusion_matrix_pred_aug1[t.long(), p.long()] += 1
                for t, p in zip(pred_aug1.view(-1), pred_aug2.view(-1)):
                    confusion_matrix_aug1_aug2[t.long(), p.long()] += 1
                for t, p in zip(pred.view(-1), pred_aug2.view(-1)):
                    confusion_matrix_pred_aug2[t.long(), p.long()] += 1

        confusion_matrices = {'train/cm_pred': confusion_matrix,
                              'train/cm_aug1': confusion_matrix_aug1,
                              'train/cm_aug2': confusion_matrix_aug2,
                              'train/cm_pred_aug1': confusion_matrix_pred_aug1,
                              'train/cm_pred_aug2': confusion_matrix_pred_aug2,
                              'train/cm_aug1_aug2': confusion_matrix_aug1_aug2}
        # logging total results
        # features
        denom = math.floor(len(data_loader.dataset) / self.args.batch_size)
        for key, value in wandb_features.items():
            wandb_features[key] = wandb_features[key] / denom
        wandb_features['p_clean_sample'] = feature['p_clean']
        wandb_features['p_aug1_sample'] = feature['p_aug1']

        # loss
        denom = len(data_loader.dataset) / self.args.batch_size
        test_loss = total_loss / denom

        # error
        datasize = len(data_loader.dataset)
        test_acc = total_correct / datasize

        return test_loss, test_acc, wandb_features, confusion_matrices


    def test_c_save(self, test_dataset, base_path=None):
        """Evaluate network on given corrupted dataset."""
        wandb_features, wandb_plts = dict(), dict()
        wandb_table = pd.DataFrame(columns=CORRUPTIONS, index=['loss', 'error'])
        confusion_matrices = []
        if (self.args.dataset == 'cifar10') or (self.args.dataset == 'cifar100'):
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

                test_loss, test_acc, _, confusion_matrix = self.test_save(test_loader, data_type=corruption)

                wandb_table[corruption]['loss'] = test_loss
                wandb_table[corruption]['error'] = 100 - 100. * test_acc
                # wandb_plts[corruption] = confusion_matrix

                corruption_accs.append(test_acc)
                confusion_matrices.append(confusion_matrix)
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    corruption, test_loss, 100 - 100. * test_acc))

            # return np.mean(corruption_accs), wandb_features
            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm
        else:  # imagenet
            corruption_accs = []

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            preprocess = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess,
            ])

            for c in CORRUPTIONS:
                print(c)
                severity_accs = []
                severity_losses = []
                for s in range(1, 6):
                    # valdir = os.path.join(self.args.corrupted_data, c, str(s))
                    valdir = os.path.join(base_path, c, str(s))
                    test_dataset = datasets.ImageFolder(valdir, test_transform)
                    val_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.args.eval_batch_size,
                        shuffle=False,
                        num_workers=self.args.num_workers,
                        pin_memory=True)

                    loss, acc1, _, confusion_matrix = self.test(val_loader)
                    severity_accs.append(acc1)
                    severity_losses.append(loss)

                test_loss = np.mean(severity_losses)
                test_acc = np.mean(severity_accs)
                wandb_table[c]['loss'] = test_loss
                wandb_table[c]['error'] = 100 - 100. * test_acc

                corruption_accs.append(test_acc)
                confusion_matrices.append(confusion_matrix)
                print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
                    c, test_loss, 100 - 100. * test_acc))

            test_c_acc = np.mean(corruption_accs)
            test_c_cm = np.mean(confusion_matrices, axis=0)
            return test_c_acc, wandb_table, test_c_cm


    def test_save(self, data_loader, data_type='clean'):
        """Evaluate network on given dataset."""
        self.net.eval()
        total_loss, total_correct = 0., 0.
        wandb_features = dict()
        confusion_matrix = torch.zeros(self.classes, self.classes)
        tsne_features = []
        B = self.args.eval_batch_size
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = self.net(images)

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

                # save false examples
                inds = (pred != targets).nonzero(as_tuple=True)[0]
                inds = inds.detach().cpu().numpy()
                images = images * 0.5 + 0.5

                for ind in inds:
                    folderpath = os.path.join(self.args.save, 'false_images')
                    if not os.path.exists(folderpath):
                        os.mkdir(folderpath)
                    f = (i * B + ind) % 10000
                    s = (i * B + ind) // 10000 + 1
                    t = targets[ind]
                    p = pred[ind]
                    filename = f"f{f}_{data_type}_s{s}_t{t}_p{p}.png"
                    filepath = os.path.join(folderpath, filename)

                    image = images[ind].detach().cpu().numpy()
                    image = np.transpose(image, axes=[1, 2, 0])

                    plt.imsave(filepath, image)


                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                if self.args.debug == True:
                    print("debug test epoch is terminated")
                    break

        datasize = len(data_loader.dataset)
        wandb_features['test/loss'] = total_loss / datasize
        wandb_features['test/error'] = 100 - 100. * total_correct / datasize
        test_loss = total_loss / datasize
        test_acc = total_correct / datasize
        return test_loss, test_acc, wandb_features, confusion_matrix