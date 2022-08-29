import torch
import torch.nn.functional as F
import time

from losses import get_additional_loss, CenterLoss


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
          correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
          res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

        datasize = len(data_loader.dataset)
        wandb_features['train/ce_loss'] = total_ce_loss / datasize
        wandb_features['train/additional_loss'] = total_additional_loss / datasize
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / datasize
        wandb_features['train/error'] = 100 - 100. * total_correct / datasize
        return loss_ema, wandb_features

    def train(self, data_loader):
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

                logits_all = self.net(images_all, targets)
                self.wandb_input = self.net.get_wandb_input()

                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                ce_loss = F.cross_entropy(logits_clean, targets)
                additional_loss = get_additional_loss(self.args.additional_loss,
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
                # acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()

            batch_ema = 0.1 * batch_ema + 0.9 * float(batch_time)
            data_ema = 0.1 * data_ema + 0.9 * float(data_time)
            loss_ema = 0.9 * loss_ema + 0.1 * float(loss)

            if i % self.args.print_freq == 0:
                print(
                    'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
                    '{:.3f} | Train Acc5 {:.3f}'.format(i, len(data_loader), data_ema,
                                                        batch_ema, loss_ema, acc1_ema,
                                                        acc5_ema))
            if self.wandb_logger is not None:
                self.wandb_logger.after_train_iter(self.wandb_input)

        wandb_features['train/ce_loss'] = total_ce_loss / len(data_loader.dataset)
        wandb_features['train/additional_loss'] = total_additional_loss / len(data_loader.dataset)
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(data_loader.dataset)
        wandb_features['train/error'] = 100 - 100. * total_correct / len(data_loader.dataset)

        return loss_ema, wandb_features # acc1_ema, batch_ema