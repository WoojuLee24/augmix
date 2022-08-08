import torch
import torch.nn.functional as F

from losses import get_additional_loss, CenterLoss


def train(net, train_loader, args, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    wandb_features = {}
    total_ce_loss = 0.
    total_additional_loss = 0.
    total_correct = 0.
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.no_jsd or args.aug == 'none':
            # no apply additional loss. augmentations are optional
            # aug choices = ['none', 'augmix',..]
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_ce_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

        else:
            # apply additional loss
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all, targets)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
            pred = logits_clean.data.max(1)[1]

            # Cross-entropy is only computed on clean images
            ce_loss = F.cross_entropy(logits_clean, targets)
            additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
                                                  12, targets)

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

    wandb_features['train/ce_loss'] = total_ce_loss / len(train_loader.dataset)
    wandb_features['train/additional_loss'] = total_additional_loss / len(train_loader.dataset)
    wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(train_loader.dataset)
    wandb_features['train/error'] = 100 - 100. * total_correct / len(train_loader.dataset)
    return loss_ema, wandb_features


def train2(net, train_loader, args, criterion_al, optimizer, optimizer_al, scheduler):
    """Train for one epoch."""
    net.train()
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
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_ce_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

        else:
            # apply additional loss
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
            pred = logits_clean.data.max(1)[1]

            # Cross-entropy is only computed on clean images
            ce_loss = F.cross_entropy(logits_clean, targets)
            additional_loss = criterion_al(net.module.features, targets)
            # additional_loss = get_additional_loss(args.additional_loss, logits_clean, logits_aug1, logits_aug2,
            #                                       12, targets)

            loss = ce_loss + additional_loss
            total_ce_loss += float(ce_loss.data)
            total_additional_loss += float(additional_loss.data)
            total_correct += pred.eq(targets.data).sum().item()

        loss.backward()
        optimizer.step()
        # by doing so, weight_cent would not impact on the learning of centers\
        weight_cent = 1
        for param in criterion_al.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_al.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        if i % args.print_freq == 0:
            print('Train Loss {:.3f}'.format(loss_ema))

    wandb_features['train/ce_loss'] = total_ce_loss / len(train_loader.dataset)
    wandb_features['train/additional_loss'] = total_additional_loss / len(train_loader.dataset)
    wandb_features['train/loss'] = (total_ce_loss + total_additional_loss) / len(train_loader.dataset)
    wandb_features['train/error'] = 100 - 100. * total_correct / len(train_loader.dataset)
    return loss_ema, wandb_features