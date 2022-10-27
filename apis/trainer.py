import torch
import torch.nn.functional as F
import time
import os
import math

from torchvision import transforms
from losses import get_additional_loss, CenterLoss
from datasets.APR import mix_data

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


class Trainer():
    def __init__(self, net, args, optimizer, scheduler, wandb_logger=None, device='cuda', additional_loss=None):
        self.net = net
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.wandb_logger = wandb_logger
        self.wandb_input = dict()
        self.device = device
        self.additional_loss = additional_loss

    def __call__(self, data_loader):
        if self.args.additional_loss in ['center_loss', 'mlpjsd']:
            train_loss_ema, train_features = self.train2(data_loader)
        elif self.args.additional_loss in ['mlpjsdv1.1']:
            train_loss_ema, train_features = self.train1_1(data_loader, self.args, self.optimizer, self.scheduler)
        else:
            train_loss_ema, train_features = self.train(data_loader)
        return train_loss_ema, train_features

    def train1_1(self, data_loader):
        """Train for one epoch."""
        self.net.train()
        wandb_features = {}
        total_ce_loss, total_additional_loss, total_correct, loss_ema = 0., 0., 0., 0.
        for i, (images, targets) in enumerate(data_loader):
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()

            self.optimizer.zero_grad()
            self.net.module.hook_features.clear()

            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            else:
                # apply additional loss
                images_all = torch.cat(images, 0).cuda()
                targets = targets.cuda()

                logits_all = self.net(images_all, targets)
                self.wandb_input = self.net.get_wandb_input()
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]

                # Cross-entropy is only computed on clean images
                ce_loss = F.cross_entropy(logits_clean, targets)
                # additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
                #                                       12, targets, args.temper)
                features_clean, features_aug1, features_aug2 = torch.split(self.net.module.feature_projected, images[0].size(0))
                additional_loss = get_additional_loss(self.args.additional_loss, features_clean, features_aug1, features_aug2,
                                                      12, targets, self.args.temper)
                for key, feature in self.net.module.hook_features.items():
                    feature_clean, feature_aug1, feature_aug2 = torch.split(feature[0], images[0].size(0))
                    k = get_additional_loss(self.args.additional_loss, feature_clean, feature_aug1, feature_aug2,
                                                      12, targets, self.args.temper, self.args.reduction)
                    additional_loss += k

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            if i % self.args.print_freq == 0:
                print('Train Loss {:.3f}'.format(loss_ema))

            if self.wandb_logger is not None:
                self.wandb_logger.after_train_iter(self.wandb_input)

        datasize = len(data_loader.dataset)
        wandb_features['train/ce_loss'] = total_ce_loss / datasize
        wandb_features['train/additional_loss'] = total_additional_loss / datasize
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / datasize
        wandb_features['train/error'] = 100 - 100. * total_correct / datasize
        return loss_ema, wandb_features

    def train2(self, data_loader):
        """Train for one epoch."""
        self.net.train()
        wandb_features = {}
        total_ce_loss, total_additional_loss, total_correct, loss_ema = 0., 0., 0., 0.
        for i, (images, targets) in enumerate(data_loader):
            self.optimizer.zero_grad()
            if self.additional_loss:
                self.additional_loss['optimizer_al'].zero_grad()

            if self.args.no_jsd or self.args.aug == 'none':
                images, targets = images.to(self.device), targets.to(self.device)

                logits = self.net(images)
                self.wandb_input = self.net.get_wandb_input()

                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                total_ce_loss += float(loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            else:
                # apply additional loss
                images_all = torch.cat(images, 0).cuda()
                targets = targets.cuda()

                logits_all = self.net(images_all)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]

                # Cross-entropy is only computed on clean images
                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss = 0.
                if self.additional_loss:
                    additional_loss = self.additional_loss['criterion_al'](
                        self.args, logits_all, self.net.module.features, targets)

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.additional_loss:
                self.additional_loss['optimizer_al'].step()
                self.additional_loss['scheduler_al'].step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            if i % self.args.print_freq == 0:
                print('Train Loss {:.3f}'.format(loss_ema))
            if self.wandb_logger is not None:
                self.wandb_logger.after_train_iter(self.wandb_input)

        datasize = len(data_loader.dataset)
        wandb_features['train/ce_loss'] = total_ce_loss / datasize
        wandb_features['train/additional_loss'] = total_additional_loss / datasize
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / datasize
        wandb_features['train/error'] = 100 - 100. * total_correct / datasize
        return loss_ema, wandb_features

    def train3(self, data_loader, epoch=0):
        """Train for one epoch."""
        """
        log jsd distance of same instance, same class, and different class.
        use jsdv3 additional_loss
        """
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
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

        # logging total results
        denom = math.floor(len(data_loader.dataset) / self.args.batch_size)
        # features
        for key, value in wandb_features.items():
            wandb_features[key] = wandb_features[key] / denom
        wandb_features['train/p_clean_sample'] = feature['p_clean']
        wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        return loss_ema, wandb_features  # acc1_ema, batch_ema

    def train3_simsiam(self, data_loader, epoch=0):
        """Train for one epoch."""
        """
        log jsd distance of same instance, same class, and different class.
        use jsdv3 additional_loss
        """
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
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
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                orig, aug1, aug2 = self.net(images_all)  # , targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, z_clean = orig
                logits_aug1, z_aug1 = aug1
                logits_aug2, z_aug2 = aug2

                pred = logits_clean.data.max(1)[1]
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug2.data.max(1)[1]

                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss, feature = get_additional_loss(self.args,
                                                               orig, aug1, aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)

                if self.args.debug == True:
                    B = images[0].size(0)
                    pred_aug_error = 100 - 100 * (
                                pred_aug1.eq(pred.data).sum().item() + pred_aug2.eq(pred.data).sum().item()) / 2 / B

                    if torch.isnan(additional_loss):
                        print('pred_aug_error: ', pred_aug_error)
                        print('prev_feature: ', prev_feature)
                        save_path = os.path.join(self.args.save,
                                                 'epoch{}_pred_aug_error{}_checkpoint.pth.tar'.format(epoch,
                                                                                                      pred_aug_error))
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
        wandb_features['train/p_clean_sample'] = feature['p_clean']
        wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        return loss_ema, wandb_features  # acc1_ema, batch_ema

    # when using apr_p without jsdloss
    def train_apr_p(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss, total_correct, = 0., 0., 0.
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
                if self.args.apr_p == 1:
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

            else:
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all) #, targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss = get_additional_loss(self.args,
                                                      logits_clean, logits_aug1, logits_aug2,
                                                      self.args.lambda_weight, targets, self.args.temper,
                                                      self.args.reduction)
                for key, feature in self.net.module.hook_features.items():
                    feature_clean, feature_aug1, feature_aug2 = torch.split(feature[0], images[0].size(0))
                    additional_loss += get_additional_loss(self.args.additional_loss,
                                                           feature_clean, feature_aug1, feature_aug2,
                                                           self.args.lambda_weight, targets, self.args.temper,
                                                           self.args.reduction)

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                pred = logits_clean.data.max(1)[1]
                total_correct += pred.eq(targets.data).sum().item()

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

        wandb_features['train/ce_loss'] = total_ce_loss / len(data_loader.dataset)
        wandb_features['train/additional_loss'] = total_additional_loss / len(data_loader.dataset)
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(data_loader.dataset)
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)

        return loss_ema, wandb_features # acc1_ema, batch_ema

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

    def train_auxbn(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
        total_correct, total_pred_aug_correct, total_aug_correct = 0., 0., 0.
        data_ema, batch_ema, loss_ema, acc1_ema, acc5_ema = 0., 0., 0., 0., 0.

        self.net.apply(to_mix_status)

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
                total_correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
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
                for key, feature in self.net.module.hook_features.items():
                    feature_clean, feature_aug1, feature_aug2 = torch.split(feature[0], images[0].size(0))
                    additional_loss += get_additional_loss(self.args.additional_loss,
                                                           feature_clean, feature_aug1, feature_aug2,
                                                           self.args.lambda_weight, targets, self.args.temper,
                                                           self.args.reduction)

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
        wandb_features['train/p_clean_sample'] = feature['p_clean']
        wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        return loss_ema, wandb_features  # acc1_ema, batch_ema

    def train(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
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
                total_additional_loss = 0.
                total_correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all) #, targets)
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
                for key, feature in self.net.module.hook_features.items():
                    feature_clean, feature_aug1, feature_aug2 = torch.split(feature[0], images[0].size(0))
                    additional_loss += get_additional_loss(self.args.additional_loss,
                                                           feature_clean, feature_aug1, feature_aug2,
                                                           self.args.lambda_weight, targets, self.args.temper,
                                                           self.args.reduction)

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
        denom = math.floor(len(data_loader.dataset) / self.args.batch_size)
        # features
        for key, value in wandb_features.items():
            wandb_features[key] = wandb_features[key] / denom
        wandb_features['train/p_clean_sample'] = feature['p_clean']
        wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        return loss_ema, wandb_features  # acc1_ema, batch_ema



    def train_expand(self, data_loader):
        self.net.train()
        wandb_features = dict()
        total_ce_loss, total_additional_loss = 0., 0.
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
                total_additional_loss = 0.
                total_correct += pred.eq(targets.data).sum().item()
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            else:
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)

                logits_all = self.net(images_all)  # , targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                pred = logits_clean.data.max(1)[1]
                pred_aug1 = logits_aug1.data.max(1)[1]
                pred_aug2 = logits_aug2.data.max(1)[1]

                ce_loss = F.cross_entropy(logits_clean, targets)

                prelogits_clean, prelogits_aug1, prelogits_aug2 = torch.split(self.net.module.prelogits, images[0].size(0))
                additional_loss, feature = get_additional_loss(self.args,
                                                               prelogits_clean, prelogits_aug1, prelogits_aug2,
                                                               self.args.lambda_weight, targets, self.args.temper,
                                                               self.args.reduction)

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
        wandb_features['train/p_clean_sample'] = feature['p_clean']
        wandb_features['train/p_aug1_sample'] = feature['p_aug1']

        denom = len(data_loader.dataset) / self.args.batch_size
        # loss
        wandb_features['train/ce_loss'] = total_ce_loss / denom
        wandb_features['train/additional_loss'] = total_additional_loss / denom
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / denom

        # error
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)
        wandb_features['train/aug_error'] = 100 - 100. * total_aug_correct / len(data_loader.dataset)
        wandb_features['train/pred_aug_error'] = 100 - 100. * total_pred_aug_correct / len(data_loader.dataset)

        return loss_ema, wandb_features  # acc1_ema, batch_ema



