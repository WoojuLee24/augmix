import torch
from models.builder import build_net, build_loss, get_lr

class AdditionalLoss():
    def __init__(self, args, num_classes, data_loader):
        self.criterion = build_loss(args, num_classes, data_loader)
        if self.criterion is not None:
            self.optimizer = torch.optim.SGD(self.criterion.parameters(), lr=args.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: get_lr(
                    step, # pylint: disable=g-long-lambda
                    args.epochs * len(data_loader),
                    1, # lr_lambda computes multiplicative factor
                    1e-6 / args.learning_rate))

