import torch
import torch.nn.functional as F
import time
import os
import math
import numpy as np

from torchvision import transforms
from losses import get_additional_loss, get_additional_loss2
from datasets.APR import mix_data
import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:min(k, maxk)].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

from functools import partial
def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


def imsave(image, folder, name):
    root = f"/ws/data/log/cifar10_debug/{folder}"
    os.makedirs(root, exist_ok=True)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.cpu().detach().numpy()
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    plt.imsave(root + f'/{name}.png', np.asarray(image))


class Trainer():
    def __init__(self, net, args, optimizer, scheduler, wandb_logger=None, device='cuda'):
        self.net = net
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.wandb_logger = wandb_logger
        self.wandb_input = dict()
        self.device = device

        if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
            self.classes = 10
        elif args.dataset == 'imagenet':
            self.classes = 1000
        elif args.dataset == 'imagenet100':
            self.classes = 100

    def __call__(self, data_loader):
        if self.args.additional_loss in ['center_loss', 'mlpjsd']:
            train_loss_ema, train_features = self.train2(data_loader)
        elif self.args.additional_loss in ['mlpjsdv1.1']:
            train_loss_ema, train_features = self.train1_1(data_loader, self.args, self.optimizer, self.scheduler)
        else:
            train_loss_ema, train_features = self.train(data_loader)
        return train_loss_ema, train_features


    def debug_images(self, images, title='ori'):
        import torchvision
        import matplotlib.pyplot as plt
        import numpy as np
        denormalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0., 0., 0.], [1 / 0.5, 1 / 0.5, 1 / 0.5]),
            torchvision.transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.])
        ])
        images = denormalize(images)

        images = images.cpu().detach().numpy()
        B, C, H, W = np.shape(images)
        for i in range(B):
            img = np.transpose(images[i], (1, 2, 0))
            img = np.clip(img, 0., 1.)
            plt.imsave(f'/ws/data/log/cifar10/debug_images/{title}_i{i}.png', img)


    def train(self, data_loader):
        """
        original augmix version. 0522cc
        """
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
        ce_loss_, jsd_distance, hook_distance = 0., 0., 0.
        hfeatures = {'jsd_distance': 0, 'jsd_distance_diff_class': 0, 'jsd_distance_same_class': 0, 'triplet_loss': 0}
        correct = 0.
        correct_aug1, correct_aug2, correct_orig_aug1, correct_orig_aug2, correct_aug1_aug2 = 0., 0., 0., 0., 0.
        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.

        lr = self.scheduler.get_lr()
        end = time.time()
        for i, (images, targets) in enumerate(data_loader):
            ''' Compute data loading time '''
            data_time = time.time() - end
            self.optimizer.zero_grad()
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()
            self.net.module.hook_features.clear()
            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_additional_loss = 0.
                ce_loss_ += float(loss.data)
                correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                for i in range(len(images)):
                    for k in range(10):
                        imsave(images[i][k], 'pixmix', f'{i}_{k}')
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)
                logits_all = self.net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, 1,
                                                               self.args.reduction)

                # hook_loss = 0
                # hfeature = {}
                # # hook loss
                # for hkey, hfeature in self.net.module.hook_features.items():
                #     feature_clean, feature_aug1, feature_aug2 = torch.chunk(hfeature[0], 3)
                #     feature_clean, feature_aug1, feature_aug2 = feature_clean, feature_aug1, feature_aug2,
                #     if self.args.additional_loss2 == 'ssim':
                #         _, hfeature = get_additional_loss2(self.args,
                #                                                    feature_clean,
                #                                                    feature_aug1,
                #                                                    feature_aug2,
                #                                                    self.args.aux_hlambda)
                #     else:
                #         B, C, _, _ = feature_clean.size()
                #         feature_clean, feature_aug1, feature_aug2 = feature_clean.view(B, -1), \
                #                                                     feature_aug1.view(B, -1), \
                #                                                     feature_aug2.view(B, -1)
                #         # if multi hook layer -> have to be fixed.
                #         _, hfeature = get_additional_loss2(self.args,
                #                                                    feature_clean,
                #                                                    feature_aug1,
                #                                                    feature_aug2,
                #                                                    self.args.lambda_weight2,
                #                                                    targets)

                loss = ce_loss + additional_loss

                # logging loss and distance
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                ce_loss_ += float(ce_loss.data)
                jsd_distance += feature['jsd_distance'].detach()
                # for key, value in hfeature.items():
                #     if key in hfeatures.keys():
                #         hfeatures[key] += value.detach()

                # logging error
                self.wandb_input = self.net.get_wandb_input()
                pred = logits_clean.data.max(1)[1]
                correct += pred.eq(targets.data).sum().item()
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug1.data.max(1)[1]
                correct_aug1 += pred_aug1.eq(targets.data).sum().item()
                correct_aug2 += pred_aug2.eq(targets.data).sum().item()
                correct_orig_aug1 += pred.eq(pred_aug1.data).sum().item()
                correct_aug1_aug2 += pred_aug1.eq(pred_aug2.data).sum().item()
                correct_orig_aug2 += pred.eq(pred_aug2.data).sum().item()

                acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            beta = 0.1 # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
            batch_ema = beta * batch_ema + (1-beta) * float(batch_time)
            data_ema = beta * data_ema + (1-beta) * float(data_time)
            loss_ema = beta * loss_ema + (1-beta) * float(loss)
            acc1_ema = beta * acc1_ema + (1-beta) * float(acc1)
            acc5_ema = beta * acc5_ema + (1-beta) * float(acc5)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if i % self.args.log_freq == 0:
                self.wandb_input['loss'] = float(loss)
                self.wandb_input['acc1'] = float(acc1)
                self.wandb_input['acc5'] = float(acc5)
                if self.wandb_logger is not None:
                    self.wandb_logger.after_train_iter(self.wandb_input)

        # logging total results
        denom = len(data_loader.dataset) / self.args.batch_size
        # loss with multiplied parameters
        wandb_features['train/total_ce_loss'] = total_ce_loss / denom
        wandb_features['train/total_additional_loss'] = total_additional_loss / denom
        wandb_features['train/total_loss'] = (total_ce_loss + total_additional_loss) / denom
        # loss
        wandb_features['train/ce_loss'] = ce_loss_ / denom
        # jsd distance
        wandb_features['train/jsd_distance'] = jsd_distance / denom
        # # hook distance_aux
        # wandb_features['train/hook_distance_aux'] = hook_distance / denom
        # for key, value in hfeatures.items():
        #     wandb_features[f'train/{key}'] = value
        # error
        wandb_features['train/error'] = 100 - 100. * correct / len(data_loader.dataset)
        wandb_features['train/error_aug1'] = 100 - 100. * correct_aug1 / len(data_loader.dataset)
        wandb_features['train/error_aug2'] = 100 - 100. * correct_aug1 / len(data_loader.dataset)
        wandb_features['train/error_orig_aug1'] = 100 - 100. * correct_orig_aug1 / len(data_loader.dataset)
        wandb_features['train/error_aug1_aug2'] = 100 - 100. * correct_aug1_aug2 / len(data_loader.dataset)
        wandb_features['train/error_orig_aug2'] = 100 - 100. * correct_orig_aug2 / len(data_loader.dataset)

        # lr
        wandb_features['lr'] = float(lr[0])

        train_cms = {}

        return loss_ema, wandb_features, train_cms  # acc1_ema, batch_ema


    # when using apr_p without jsdloss
    def train_apr_p(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
        total_correct, total_pred_aug_correct, total_aug_correct = 0., 0., 0.
        confusion_matrix = torch.zeros(self.classes, self.classes)
        confusion_matrix_aug1 = torch.zeros(self.classes, self.classes)
        confusion_matrix_pred_aug1 = torch.zeros(self.classes, self.classes)
        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.
        lr = self.scheduler.get_lr()
        end = time.time()

        for i, (images, targets) in enumerate(data_loader):
            ''' Compute data loading time '''
            data_time = time.time() - end
            self.optimizer.zero_grad()
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()
            self.net.module.hook_features.clear()
            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)
                if self.args.apr_p == True:
                    inputs_mix = mix_data(images, self.args.apr_mixed_coefficient)
                    inputs, inputs_mix = transforms.Normalize([0.5] * 3, [0.5] * 3)(images), transforms.Normalize([0.5] * 3,[0.5] * 3)(inputs_mix)
                    images = torch.cat([inputs, inputs_mix], dim=0)
                    targets = torch.cat([targets, targets], dim=0)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            else:
                if self.args.apr_p == True:
                    inputs = images[0]
                    inputs_mix1 = mix_data(images[1], self.args.apr_mixed_coefficient)  # default apr_mixed_coefficient 0.6
                    inputs_mix2 = mix_data(images[2], self.args.apr_mixed_coefficient)
                    inputs, inputs_mixx1, inputs_mixx2 = transforms.Normalize([0.5] * 3, [0.5] * 3)(inputs), \
                                                         transforms.Normalize([0.5] * 3, [0.5] * 3)(inputs_mix1), \
                                                         transforms.Normalize([0.5] * 3, [0.5] * 3)(inputs_mix2)
                    images = (images[0], inputs_mixx1, inputs_mixx2)

                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all)  # , targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug2.data.max(1)[1]

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)

                for hkey, hfeature in self.net.module.hook_features.items():
                    B = images[0].size(0)
                    feature_clean, feature_aug1, feature_aug2 = torch.split(hfeature[0], images[0].size(0))
                    feature_clean, feature_aug1, feature_aug2 = feature_clean.view(B, -1), \
                                                                feature_aug1.view(B, -1), \
                                                                feature_aug2.view(B, -1)
                    hook_additional_loss, hook_feature = get_additional_loss(self.args,
                                                                             feature_clean, feature_aug1, feature_aug2,
                                                                             self.args.lambda_weight, targets,
                                                                             self.args.temper,
                                                                             self.args.reduction)
                    for key, value in hook_feature.items():
                        new_key = f'{hkey}_{key}'
                        feature[new_key] = value.detach()

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)

                if i == 0:
                    for key, value in feature.items():
                        total_key = 'train/total_' + key
                        wandb_features[total_key] = feature[key].detach()
                else:
                    # exclude terminal data for wandb_features: batch size is different.
                    if logits_clean.size(0) == self.args.batch_size:
                        for key, value in feature.items():
                            total_key = 'train/total_' + key
                            wandb_features[total_key] += feature[key].detach()

                total_correct += pred.eq(targets.data).sum().item()
                total_pred_aug_correct += (pred_aug1.eq(pred.data).sum().item() + pred_aug2.eq(
                    pred.data).sum().item()) / 2
                total_aug_correct += (pred_aug1.eq(targets.data).sum().item() + pred_aug2.eq(
                    targets.data).sum().item()) / 2
                acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

                for t, p in zip(targets.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(targets.view(-1), pred_aug1.view(-1)):
                    confusion_matrix_aug1[t.long(), p.long()] += 1
                for t, p in zip(pred.view(-1), pred_aug1.view(-1)):
                    confusion_matrix_pred_aug1[t.long(), p.long()] += 1

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            beta = 0.1  # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
            batch_ema = beta * batch_ema + (1 - beta) * float(batch_time)
            data_ema = beta * data_ema + (1 - beta) * float(data_time)
            loss_ema = beta * loss_ema + (1 - beta) * float(loss)
            acc1_ema = beta * acc1_ema + (1 - beta) * float(acc1)
            acc5_ema = beta * acc5_ema + (1 - beta) * float(acc5)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if i % self.args.log_freq == 0:
                self.wandb_input['loss'] = float(loss)
                self.wandb_input['acc1'] = float(acc1)
                self.wandb_input['acc5'] = float(acc5)
                if self.wandb_logger is not None:
                    self.wandb_logger.after_train_iter(self.wandb_input)

            # logging total results
        denom = math.floor(len(data_loader.dataset) / self.args.batch_size)
        # features
        for key, value in wandb_features.items():
            wandb_features[key] = wandb_features[key] / denom
        # wandb_features['train/p_clean_sample'] = feature['p_clean']
        # wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        # lr
        wandb_features['lr'] = float(lr[0])

        # confusion_matrices
        train_cms = {'train/cm_pred': confusion_matrix.detach().cpu().numpy(),
                     'train/cm_aug1': confusion_matrix_aug1.detach().cpu().numpy(),
                     'train/cm_pred_aug1': confusion_matrix_pred_aug1.detach().cpu().numpy()}

        return loss_ema, wandb_features, train_cms  # acc1_ema, batch_ema

    # when using apr_p with jsdloss
    def train3_apr_p(self, data_loader, epoch=0):
        """Train for one epoch."""
        """
        log jsd distance of same instance, same class, and different class.
        use jsdv3 additional_loss
        """
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss, total_triplet_loss, \
        total_same_instance_loss, total_same_class_loss, total_diff_class_loss, = 0., 0., 0., 0., 0., 0.
        total_correct, total_pred_aug_correct, total_aug_correct = 0., 0., 0.
        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.

        end = time.time()
        for i, (images, targets) in enumerate(data_loader):
            ''' Compute data loading time '''
            data_time = time.time() - end
            self.optimizer.zero_grad()
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()
            self.net.module.hook_features.clear()

            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                if self.args.apr_p == 1:
                    inputs_mix1 = mix_data(images[1], self.args.apr_mixed_coefficient)
                    inputs_mix2 = mix_data(images[2], self.args.apr_mixed_coefficient)
                    inputs_mixx1, inputs_mixx2 = transforms.Normalize([0.5] * 3, [0.5] * 3)(inputs_mix1), transforms.Normalize([0.5] * 3, [0.5] * 3)(inputs_mix2)
                    images = (images[0], inputs_mixx1, inputs_mixx2)

                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all)  # , targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug2.data.max(1)[1]

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)
                if self.args.debug == True:
                    B = images[0].size(0)
                    pred_aug_error = 100 - 100 * (pred_aug1.eq(pred.data).sum().item() + pred_aug2.eq(pred.data).sum().item()) / 2 / B

                    if torch.isnan(additional_loss):
                        print('pred_aug_error: ', pred_aug_error)
                        print('prev_feature: ', prev_feature)
                        save_path = os.path.join(self.args.save, 'epoch{}_pred_aug_error{}_checkpoint.pth.tar'.format(epoch, pred_aug_error))
                        checkpoint = {
                            'epoch': epoch,
                            'best_acc': 0,
                            'state_dict': self.net.state_dict(),
                            'additional_loss': additional_loss,
                            'optimizer': self.optimizer.state_dict(),
                            'feature': feature
                        }
                        torch.save(checkpoint, save_path)


                    else:
                        prev_feature = feature

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                total_same_instance_loss += float(feature['jsd_distance'])
                total_same_class_loss += float(feature['jsd_distance_same_class'])
                total_diff_class_loss += float(feature['jsd_distance_diff_class'])
                total_correct += pred.eq(targets.data).sum().item()
                total_pred_aug_correct += (pred_aug1.eq(pred.data).sum().item() + pred_aug2.eq(pred.data).sum().item()) / 2
                total_aug_correct += (pred_aug1.eq(targets.data).sum().item() + pred_aug2.eq(targets.data).sum().item()) / 2
                acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            beta = 0.1  # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
            batch_ema = beta * batch_ema + (1 - beta) * float(batch_time)
            data_ema = beta * data_ema + (1 - beta) * float(data_time)
            loss_ema = beta * loss_ema + (1 - beta) * float(loss)
            acc1_ema = beta * acc1_ema + (1 - beta) * float(acc1)
            acc5_ema = beta * acc5_ema + (1 - beta) * float(acc5)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if i % self.args.log_freq == 0:
                self.wandb_input['loss'] = float(loss)
                self.wandb_input['acc1'] = float(acc1)
                self.wandb_input['acc5'] = float(acc5)
                if self.wandb_logger is not None:
                    self.wandb_logger.after_train_iter(self.wandb_input)

        denom = len(data_loader.dataset) / self.args.batch_size
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/total_same_instance_loss'] = total_same_instance_loss / denom
        wandb_features['train/total_same_class_loss'] = total_same_class_loss / denom
        wandb_features['train/total_diff_class_loss'] = total_diff_class_loss / denom
        wandb_features['train/triplet_loss'] = total_triplet_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)
        wandb_features['train/jsd_matrix'] = feature['jsd_matrix']
        wandb_features['train/p_clean'] = feature['p_clean']
        wandb_features['train/p_aug1'] = feature['p_aug1']

        return loss_ema, wandb_features  # acc1_ema, batch_ema


    def train_augda(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
        ce_loss_, jsd_distance, jsd_distance2, hook_distance = 0., 0., 0., 0.
        correct = 0.

        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.
        lr = self.scheduler.get_lr()
        end = time.time()
        for i, (images, targets) in enumerate(data_loader):
            ''' Compute data loading time '''
            data_time = time.time() - end
            self.optimizer.zero_grad()
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()
            self.net.module.hook_features.clear()
            if self.args.no_jsd or self.args.aug == 'none':
                # temp code 0419
                if self.args.aux_type == 'unoise':
                    B, C, H, W = images.size()
                    s1 = self.args.aux_severity * torch.rand((B, 1, 1, 1))
                    unoise = 2 * torch.rand(B, C, H, W) - 1
                    images = images + s1 * unoise

                images, targets = images.to(self.device), targets.to(self.device)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_additional_loss = 0.
                ce_loss_ += float(loss.data)
                correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                # temp code 0419
                if self.args.aux_type == 'unoise':
                    B, C, H, W = images[1].size()
                    s1 = self.args.aux_severity * torch.rand((B, 1, 1, 1))
                    s2 = self.args.aux_severity * torch.rand((B, 1, 1, 1))

                    p1 = np.random.choice(B, int(B * self.args.aux_prob), replace=False)
                    p2 = np.random.choice(B, int(B * self.args.aux_prob), replace=False)
                    s1[p1] = 0
                    s2[p2] = 0

                    unoise1 = 2 * torch.rand(B, C, H, W) - 1
                    unoise2 = 2 * torch.rand(B, C, H, W) - 1
                    images[1] = images[1] + s1 * unoise1
                    images[2] = images[2] + s2 * unoise2

                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)
                logits_all = self.net(images_all)  # , targets)
                logits_clean, logits_aug1, logits_aug2, logits_aug3, logits_aug4 = torch.split(logits_all,
                                                                                               images[0].size(0))

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)
                additional_loss2, feature2 = get_additional_loss(self.args,
                                                                 logits_clean, logits_aug3, logits_aug4,
                                                                 self.args.lambda_weight2, targets, self.args.temper,
                                                                 self.args.reduction)

                # hook_loss = 0
                # # hook loss
                # for hkey, hfeature in self.net.module.hook_features.items():
                #     feature_clean, feature_aug1, feature_aug2, feature_aug3, feature_aug4 = torch.chunk(hfeature[0], 5)
                #     if self.args.additional_loss2 == 'ssim':
                #         hook_loss, hfeature = get_additional_loss2(self.args,
                #                                                    feature_clean,
                #                                                    feature_aug1,
                #                                                    feature_aug2,
                #                                                    self.args.aux_hlambda)
                #     else:
                #         feature_clean, feature_aug1, feature_aug2 = feature_clean.view(B, -1), \
                #                                                     feature_aug1.view(B, -1), \
                #                                                     feature_aug2.view(B, -1)
                #         B, C = feature_clean.size()
                #         # if multi hook layer -> have to be fixed.
                #         hook_loss, hfeature = get_additional_loss2(self.args,
                #                                                    feature_clean,
                #                                                    feature_aug1,
                #                                                    feature_aug2,
                #                                                    self.args.aux_hlambda)

                loss = ce_loss + additional_loss + additional_loss2

                # logging loss and distance
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                ce_loss_ += float(ce_loss.data)
                jsd_distance += feature['jsd_distance'].detach()
                jsd_distance2 += feature2['jsd_distance'].detach()
                # for key, value in hfeature.items():
                #     hook_distance += value.detach()

                # logging error
                self.wandb_input = self.net.get_wandb_input()
                pred = logits_clean.data.max(1)[1]
                correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            beta = 0.1  # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
            batch_ema = beta * batch_ema + (1 - beta) * float(batch_time)
            data_ema = beta * data_ema + (1 - beta) * float(data_time)
            loss_ema = beta * loss_ema + (1 - beta) * float(loss)
            acc1_ema = beta * acc1_ema + (1 - beta) * float(acc1)
            acc5_ema = beta * acc5_ema + (1 - beta) * float(acc5)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if i % self.args.log_freq == 0:
                self.wandb_input['loss'] = float(loss)
                self.wandb_input['acc1'] = float(acc1)
                self.wandb_input['acc5'] = float(acc5)
                if self.wandb_logger is not None:
                    self.wandb_logger.after_train_iter(self.wandb_input)

        # logging total results
        denom = len(data_loader.dataset) / self.args.batch_size
        # loss with multiplied parameters
        wandb_features['train/total_ce_loss'] = total_ce_loss / denom
        wandb_features['train/total_additional_loss'] = total_additional_loss / denom
        wandb_features['train/total_loss'] = (total_ce_loss + total_additional_loss) / denom
        # loss
        wandb_features['train/ce_loss'] = ce_loss_ / denom
        # jsd distance
        wandb_features['train/jsd_distance'] = jsd_distance / denom
        wandb_features['train/jsd_distance2'] = jsd_distance2 / denom
        # hook distance_aux
        wandb_features['train/hook_distance_aux'] = hook_distance / denom
        # error
        wandb_features['train/error'] = 100 - 100. * correct / len(data_loader.dataset)
        # lr
        wandb_features['lr'] = float(lr[0])

        train_cms = {}

        return loss_ema, wandb_features, train_cms  # acc1_ema, batch_ema

    # def train_prime(self, data_loader, prime_module):
    #     self.net.train()
    #     wandb_features = dict()
    #     additional_loss, hook_additional_loss = 0., torch.tensor(0.)
    #     total_ce_loss, total_additional_loss, total_hook_additional_loss = 0., 0., 0.
    #     total_correct, total_pred_aug_correct, total_aug_correct, total_robust = 0., 0., 0., 0.
    #     confusion_matrix = torch.zeros(self.classes, self.classes)
    #     confusion_matrix_aug1 = torch.zeros(self.classes, self.classes)
    #     confusion_matrix_pred_aug1 = torch.zeros(self.classes, self.classes)
    #     data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.
    #     lr = self.scheduler.get_lr()
    #     end = time.time()
    #     for i, (images, targets) in enumerate(data_loader):
    #         # if self.args.debug == True:
    #         #     if i == 2:
    #         #         print("debug train epoch is terminated")
    #         #         break
    #
    #         ''' Compute data loading time '''
    #         data_time = time.time() - end
    #         self.optimizer.zero_grad()
    #         if self.wandb_logger is not None:
    #             self.wandb_logger.before_train_iter()
    #         self.net.module.hook_features.clear()
    #         if self.args.no_jsd or self.args.aug == 'none':
    #             images, targets = images.to(self.device), targets.to(self.device)
    #             prime_module = prime_module.to(self.device)
    #             prime_module.no_jsd = True
    #             images = prime_module(images)
    #
    #             logits = self.net(images)
    #             self.wandb_input = self.net.get_wandb_input()
    #
    #             loss = F.cross_entropy(logits, targets)
    #             pred = logits.data.max(1)[1]
    #             total_ce_loss += float(loss.data)
    #             total_additional_loss = 0.
    #             total_correct += pred.eq(targets.data).sum().item()
    #             acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
    #
    #         else:
    #
    #             images = images.to(self.device)
    #             prime_module = prime_module.to(self.device)
    #             images_all = prime_module(images)
    #             # images_all = torch.cat(images, 0).to(self.device)
    #             targets = targets.to(self.device)
    #             logits_all = self.net(images_all) #, targets)
    #             logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images.size(0))
    #
    #             pred = logits_clean.data.max(1)[1]
    #             pred_aug1 = logits_aug1.data.max(1)[1]
    #             pred_aug2 = logits_aug2.data.max(1)[1]
    #
    #             ce_loss = F.cross_entropy(logits_clean, targets)
    #             additional_loss, feature = get_additional_loss(self.args,
    #                                                            logits_clean, logits_aug1, logits_aug2,
    #                                                            self.args.lambda_weight, targets, self.args.temper,
    #                                                            self.args.reduction)
    #
    #             self.wandb_input = self.net.get_wandb_input()
    #
    #             for hkey, hfeature in self.net.module.hook_features.items():
    #                 B = images[0].size(0)
    #                 feature_clean, feature_aug1, feature_aug2 = torch.split(hfeature[0], images[0].size(0))
    #                 feature_clean, feature_aug1, feature_aug2 = feature_clean.view(B, -1), feature_aug1.view(B, -1), feature_aug2.view(B, -1)
    #                 hook_additional_loss, hook_feature = get_additional_loss2(self.args,
    #                                                                           feature_clean, feature_aug1, feature_aug2,
    #                                                                           self.args.lambda_weight2, targets, self.args.temper,
    #                                                                           self.args.reduction)
    #                 if (self.args.model == 'wrnproj') and ('module.avgpool' in hkey):
    #                     hook_additional_loss = hook_additional_loss.detach()
    #                     hook_additional_loss = torch.tensor(0.).cuda()
    #
    #                 for key, value in hook_feature.items():
    #                     new_key = f'{hkey}_{key}'
    #                     feature[new_key] = value.detach()
    #
    #             loss = ce_loss + additional_loss + hook_additional_loss
    #             total_ce_loss += float(ce_loss.data)
    #             total_additional_loss += float(additional_loss.data)
    #             total_hook_additional_loss += float(hook_additional_loss.data)
    #
    #             if i == 0:
    #                 for key, value in feature.items():
    #                     total_key = 'train/total_' + key
    #                     wandb_features[total_key] = feature[key].detach()
    #             else:
    #                 # exclude terminal data for wandb_features: batch size is different.
    #                 if logits_clean.size(0) == self.args.batch_size:
    #                     for key, value in feature.items():
    #                         total_key = 'train/total_' + key
    #                         wandb_features[total_key] += feature[key].detach()
    #
    #             total_correct += pred.eq(targets.data).sum().item()
    #             total_robust += (pred.eq(targets.data) & pred_aug1.eq(pred.data) & pred_aug2.eq(pred.data)).sum().item()
    #             total_pred_aug_correct += (pred_aug1.eq(pred.data).sum().item() + pred_aug2.eq(
    #                 pred.data).sum().item()) / 2
    #             total_aug_correct += (pred_aug1.eq(targets.data).sum().item() + pred_aug2.eq(
    #                 targets.data).sum().item()) / 2
    #             acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))
    #
    #             for t, p in zip(targets.view(-1), pred.view(-1)):
    #                 confusion_matrix[t.long(), p.long()] += 1
    #             for t, p in zip(targets.view(-1), pred_aug1.view(-1)):
    #                 confusion_matrix_aug1[t.long(), p.long()] += 1
    #             for t, p in zip(pred.view(-1), pred_aug1.view(-1)):
    #                 confusion_matrix_pred_aug1[t.long(), p.long()] += 1
    #
    #         loss.backward()
    #         self.optimizer.step()
    #         self.scheduler.step()
    #
    #         batch_time = time.time() - end
    #         end = time.time()
    #
    #         beta = 0.1 # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
    #         batch_ema = beta * batch_ema + (1-beta) * float(batch_time)
    #         data_ema = beta * data_ema + (1-beta) * float(data_time)
    #         loss_ema = beta * loss_ema + (1-beta) * float(loss)
    #         acc1_ema = beta * acc1_ema + (1-beta) * float(acc1)
    #         acc5_ema = beta * acc5_ema + (1-beta) * float(acc5)
    #
    #         if i % self.args.print_freq == 0:
    #             print(
    #                 'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
    #                 '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
    #                                                     batch_ema, loss_ema, acc1_ema,
    #                                                     acc5_ema))
    #         if i % self.args.log_freq == 0:
    #             self.wandb_input['loss'] = float(loss)
    #             self.wandb_input['acc1'] = float(acc1)
    #             self.wandb_input['acc5'] = float(acc5)
    #             if self.wandb_logger is not None:
    #                 self.wandb_logger.after_train_iter(self.wandb_input)
    #
    #     # logging total results
    #     denom = math.floor(len(data_loader.dataset) / self.args.batch_size)
    #     # features
    #     for key, value in wandb_features.items():
    #         wandb_features[key] = wandb_features[key] / denom
    #     # wandb_features['train/p_clean_sample'] = feature['p_clean']
    #     # wandb_features['train/p_aug1_sample'] = feature['p_aug1']
    #
    #     denom = len(data_loader.dataset) / self.args.batch_size
    #     # loss
    #     wandb_features['train/ce_loss'] = total_ce_loss / denom
    #     wandb_features['train/additional_loss'] = total_additional_loss / denom
    #     wandb_features['train/hook_additional_loss'] = total_hook_additional_loss / denom
    #     wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom
    #
    #     # error
    #     wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
    #     wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
    #     wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)
    #     wandb_features['train/robust_error'] = 100 - 100. * total_robust / total_correct
    #
    #     # lr
    #     wandb_features['lr'] = float(lr[0])
    #
    #     # confusion_matrices
    #     train_cms = {'train/cm_pred': confusion_matrix.detach().cpu().numpy(),
    #                  'train/cm_aug1': confusion_matrix_aug1.detach().cpu().numpy(),
    #                  'train/cm_pred_aug1': confusion_matrix_pred_aug1.detach().cpu().numpy()}
    #
    #     return loss_ema, wandb_features, train_cms  # acc1_ema, batch_ema

    def train_prime(self, data_loader, prime_module):
        """
        original augmix version. 0522cc
        """
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
        ce_loss_, jsd_distance, hook_distance = 0., 0., 0.
        # hfeatures = {'jsd_distance': 0, 'jsd_distance_diff_class': 0, 'jsd_distance_same_class': 0, 'triplet_loss': 0}
        correct = 0.
        correct_aug1, correct_aug2, correct_orig_aug1, correct_orig_aug2, correct_aug1_aug2 = 0., 0., 0., 0., 0.
        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.

        lr = self.scheduler.get_lr()
        end = time.time()
        for i, (images, targets) in enumerate(data_loader):
            ''' Compute data loading time '''
            data_time = time.time() - end
            self.optimizer.zero_grad()
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()
            self.net.module.hook_features.clear()
            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)
                prime_module = prime_module.to(self.device)
                prime_module.no_jsd = True
                images = prime_module(images)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_additional_loss = 0.
                ce_loss_ += float(loss.data)
                correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                images = images.to(self.device)
                prime_module = prime_module.to(self.device)
                images_all = prime_module(images)

                # # debug
                # imgs = torch.split(images_all, images.size(0))
                # for i in range(len(imgs)):
                #     for k in range(10):
                #         imsave(imgs[i][k], "prime", f'{i}_{k}')

                # images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)
                logits_all = self.net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images.size(0))

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               logits_clean, logits_aug1, logits_aug2,
                                                               self.args.lambda_weight, targets, 1,
                                                               self.args.reduction)

                # hook_loss = 0
                # hfeature = {}
                # # hook loss
                # for hkey, hfeature in self.net.module.hook_features.items():
                #     feature_clean, feature_aug1, feature_aug2 = torch.chunk(hfeature[0], 3)
                #     feature_clean, feature_aug1, feature_aug2 = feature_clean, feature_aug1, feature_aug2,
                #     if self.args.additional_loss2 == 'ssim':
                #         _, hfeature = get_additional_loss2(self.args,
                #                                            feature_clean,
                #                                            feature_aug1,
                #                                            feature_aug2,
                #                                            self.args.aux_hlambda)
                #     else:
                #         B, C, _, _ = feature_clean.size()
                #         feature_clean, feature_aug1, feature_aug2 = feature_clean.view(B, -1), \
                #                                                     feature_aug1.view(B, -1), \
                #                                                     feature_aug2.view(B, -1)
                #         # if multi hook layer -> have to be fixed.
                #         _, hfeature = get_additional_loss2(self.args,
                #                                            feature_clean,
                #                                            feature_aug1,
                #                                            feature_aug2,
                #                                            self.args.lambda_weight2,
                #                                            targets)

                loss = ce_loss + additional_loss

                # logging loss and distance
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                ce_loss_ += float(ce_loss.data)
                jsd_distance += feature['jsd_distance'].detach()
                # for key, value in hfeature.items():
                #     if key in hfeatures.keys():
                #         hfeatures[key] += value.detach()

                # logging error
                self.wandb_input = self.net.get_wandb_input()
                pred = logits_clean.data.max(1)[1]
                correct += pred.eq(targets.data).sum().item()
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug1.data.max(1)[1]
                correct_aug1 += pred_aug1.eq(targets.data).sum().item()
                correct_aug2 += pred_aug2.eq(targets.data).sum().item()
                correct_orig_aug1 += pred.eq(pred_aug1.data).sum().item()
                correct_aug1_aug2 += pred_aug1.eq(pred_aug2.data).sum().item()
                correct_orig_aug2 += pred.eq(pred_aug2.data).sum().item()

                acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            beta = 0.1  # TODO: what is the good beta value? 0.1(noisy and fast) or 0.9(smooth and slow)?
            batch_ema = beta * batch_ema + (1 - beta) * float(batch_time)
            data_ema = beta * data_ema + (1 - beta) * float(data_time)
            loss_ema = beta * loss_ema + (1 - beta) * float(loss)
            acc1_ema = beta * acc1_ema + (1 - beta) * float(acc1)
            acc5_ema = beta * acc5_ema + (1 - beta) * float(acc5)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if i % self.args.log_freq == 0:
                self.wandb_input['loss'] = float(loss)
                self.wandb_input['acc1'] = float(acc1)
                self.wandb_input['acc5'] = float(acc5)
                if self.wandb_logger is not None:
                    self.wandb_logger.after_train_iter(self.wandb_input)

        # logging total results
        denom = len(data_loader.dataset) / self.args.batch_size
        # loss with multiplied parameters
        wandb_features['train/total_ce_loss'] = total_ce_loss / denom
        wandb_features['train/total_additional_loss'] = total_additional_loss / denom
        wandb_features['train/total_loss'] = (total_ce_loss + total_additional_loss) / denom
        # loss
        wandb_features['train/ce_loss'] = ce_loss_ / denom
        # jsd distance
        wandb_features['train/jsd_distance'] = jsd_distance / denom
        # # hook distance_aux
        # wandb_features['train/hook_distance_aux'] = hook_distance / denom
        # for key, value in hfeatures.items():
        #     wandb_features[f'train/{key}'] = value
        # error
        wandb_features['train/error'] = 100 - 100. * correct / len(data_loader.dataset)
        wandb_features['train/error_aug1'] = 100 - 100. * correct_aug1 / len(data_loader.dataset)
        wandb_features['train/error_aug2'] = 100 - 100. * correct_aug1 / len(data_loader.dataset)
        wandb_features['train/error_orig_aug1'] = 100 - 100. * correct_orig_aug1 / len(data_loader.dataset)
        wandb_features['train/error_aug1_aug2'] = 100 - 100. * correct_aug1_aug2 / len(data_loader.dataset)
        wandb_features['train/error_orig_aug2'] = 100 - 100. * correct_orig_aug2 / len(data_loader.dataset)

        # lr
        wandb_features['lr'] = float(lr[0])

        train_cms = {}

        return loss_ema, wandb_features, train_cms  # acc1_ema, batch_ema





