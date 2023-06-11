import torch
import torch.nn.functional as F
import time
import os
import math
import numpy as np

from torchvision import transforms
from aug_losses import get_aug_loss
from losses import get_ssim
from datasets.APR import mix_data
import matplotlib.pyplot as plt
from collections import defaultdict

def imsave(image, root, name):
    root = os.path.join(root, 'img')
    os.makedirs(root, exist_ok=True)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.cpu().detach().numpy()
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    plt.imsave(root + f'/{name}.png', np.asarray(image))


class AugTrainer():
    def __init__(self, net, args, optimizer, scheduler, device='cuda'):
        self.net = net
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, data_loader):
        """
        original augmix version. 0522cc
        """
        self.net.train()
        lr = self.scheduler.get_lr()
        # log
        train_features = dict()
        wandb_features = defaultdict(float) # accumulate features

        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            logits = self.net(images)
            loss, features = get_aug_loss(self.args, logits, targets)

            # log
            for key, value in features.items():
                wandb_features[key] += value

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        # log
        denom = len(data_loader.dataset) / self.args.batch_size
        for key, value in wandb_features.items():
            new_key = 'train/' + key
            train_features[new_key] = value / denom
        train_features['lr'] = lr
        print(
            f"Train Loss {train_features['train/loss']:.3f} | "
            f"Train MSE Loss {train_features['train/mse_loss']:.3f} | "
            f"Train L1 Loss {train_features['train/l1_loss']:.3f} | "
            f"Train SSIM {train_features['train/ssim']:.3f} ")

        return train_features

    def train_augop(self, data_loader):
        """
        original augmix version. 0522cc
        """
        self.net.train()
        lr = self.scheduler.get_lr()
        # log
        train_features = dict()
        wandb_features = defaultdict(float) # accumulate features

        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            logits = self.net(images)
            loss, features = get_aug_loss(self.args, logits, targets)

            # log
            for key, value in features.items():
                wandb_features[key] += value

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        # log
        denom = len(data_loader.dataset) / self.args.batch_size
        for key, value in wandb_features.items():
            new_key = 'train/' + key
            train_features[new_key] = value / denom
        train_features['lr'] = lr
        print(
            f"Train Loss {train_features['train/loss']:.3f} | "
            f"Train MSE Loss {train_features['train/mse_loss']:.3f} | "
            f"Train L1 Loss {train_features['train/l1_loss']:.3f} | "
            f"Train SSIM {train_features['train/ssim']:.3f} ")

        return train_features



class AugTester():
    def __init__(self, net, args, device='cuda'):
        self.net = net
        self.args = args
        self.device = device

    def test(self, data_loader, save_img=False):
        """Evaluate network on given dataset."""
        self.net.eval()
        # log
        test_features = dict()
        wandb_features = defaultdict(float)  # accumulate features
        self.net.module.hook_features.clear() # to prevent memory leakage
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = self.net(images)
                loss, features = get_aug_loss(self.args, logits, targets)

                # log
                for key, value in features.items():
                    wandb_features[key] += value

                if save_img == True:   # debug model aug
                    imsave(images[0], root=self.args.save, name=f"images_{i}")
                    imsave(logits[0], root=self.args.save, name=f"pred_{i}")
                    imsave(targets[0], root=self.args.save, name=f"target_{i}")

        # log
        denom = len(data_loader.dataset) / self.args.eval_batch_size
        for key, value in wandb_features.items():
            new_key = 'test/' + key
            test_features[new_key] = value / denom

        print(
            f"Test Loss {test_features['test/loss']:.3f} | "
            f"Test MSE Loss {test_features['test/mse_loss']:.3f} | "
            f"Test L1 Loss {test_features['test/l1_loss']:.3f} | "
            f"Test SSIM {test_features['test/ssim']:.3f} ")

        return test_features


    def test_augop(self, data_loader, save_img=False):
        """Evaluate network on given dataset."""
        self.net.eval()
        # log
        test_features = dict()
        wandb_features = defaultdict(float)  # accumulate features
        self.net.module.hook_features.clear() # to prevent memory leakage
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = self.net(images)
                loss, features = get_aug_loss(self.args, logits, targets)

                # log
                for key, value in features.items():
                    wandb_features[key] += value

                if save_img == True:   # debug model aug
                    image, logit, target = images[0], logits[0], targets[0]
                    logit, target = torch.split(logit, 3), torch.split(target, 3)

                    imsave(image, root=self.args.save, name=f"image{i}")
                    for j in range(len(logit)):
                        imsave(logit[j], root=self.args.save, name=f"pred{i}_aug{j}")
                        imsave(target[j], root=self.args.save, name=f"target{i}_aug{j}")

        # log
        denom = len(data_loader.dataset) / self.args.eval_batch_size
        for key, value in wandb_features.items():
            new_key = 'test/' + key
            test_features[new_key] = value / denom

        print(
            f"Test Loss {test_features['test/loss']:.3f} | "
            f"Test MSE Loss {test_features['test/mse_loss']:.3f} | "
            f"Test L1 Loss {test_features['test/l1_loss']:.3f} | "
            f"Test SSIM {test_features['test/ssim']:.3f} ")

        return test_features
