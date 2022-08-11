import torch
import torch.nn.functional as F

from losses import get_additional_loss, CenterLoss


class Trainer():
    def __init__(self, net, wandb_logger=None):
        self.wandb_logger = wandb_logger
        self.wandb_input = dict()

        self.net = net

    def train(self, train_loader, args, optimizer, scheduler):
        """Train for one epoch."""
        self.net.train()
        wandb_features = {}
        total_ce_loss = 0.
        total_additional_loss = 0.
        total_correct = 0.
        loss_ema = 0.
        for i, (images, targets) in enumerate(train_loader):
            if self.wandb_logger is not None:
                self.wandb_logger.before_train_iter()

            optimizer.zero_grad()
            self.net.module.hook_features.clear()

            if args.no_jsd or args.aug == 'none':
                # no apply additional loss. augmentations are optional
                # aug choices = ['none', 'augmix',..]
                images = images.cuda()
                targets = targets.cuda()

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
                additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
                                                      12, targets, args.temper)
                for key, feature in self.net.module.hook_features.items():
                    feature_clean, feature_aug1, feature_aug2 = torch.split(feature[0], images[0].size(0))
                    k = get_additional_loss(args.additional_loss, feature_clean, feature_aug1, feature_aug2,
                                                      12, targets, args.temper, args.reduction)
                    additional_loss += k

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            if i % args.print_freq == 0:
                print('Train Loss {:.3f}'.format(loss_ema))

            if self.wandb_logger is not None:
                self.wandb_logger.after_train_iter(self.wandb_input)

        wandb_features['train/ce_loss'] = total_ce_loss / len(train_loader.dataset)
        wandb_features['train/additional_loss'] = total_additional_loss / len(train_loader.dataset)
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(train_loader.dataset)
        wandb_features['train/error'] = 100 - 100. * total_correct / len(train_loader.dataset)
        return loss_ema, wandb_features


    def train2(self, train_loader, args, optimizer, scheduler, criterion_al, optimizer_al, scheduler_al, wandb_logger=None):
        """Train for one epoch."""
        self.net.train()
        wandb_features = {}
        total_ce_loss = 0.
        total_additional_loss = 0.
        total_correct = 0.
        loss_ema = 0.
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_al.zero_grad()

            if args.no_jsd or args.aug == 'none':
                # no apply additional loss. augmentations are optional
                # aug choices = ['none', 'augmix',..]
                images = images.cuda()
                targets = targets.cuda()

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
                additional_loss = criterion_al(args, logits_all, self.net.module.features, targets)
                # additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
                #                                       12, targets)

                loss = ce_loss + additional_loss
                total_ce_loss += float(ce_loss.data)
                total_additional_loss += float(additional_loss.data)
                total_correct += pred.eq(targets.data).sum().item()

            loss.backward()
            optimizer.step()
            optimizer_al.step()
            scheduler.step()
            scheduler_al.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            if i % args.print_freq == 0:
                print('Train Loss {:.3f}'.format(loss_ema))

        wandb_features['train/ce_loss'] = total_ce_loss / len(train_loader.dataset)
        wandb_features['train/additional_loss'] = total_additional_loss / len(train_loader.dataset)
        wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(train_loader.dataset)
        wandb_features['train/error'] = 100 - 100. * total_correct / len(train_loader.dataset)
        return loss_ema, wandb_features
