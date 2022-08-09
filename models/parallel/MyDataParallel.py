import torch


class MyDataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(MyDataParallel, self).__init__(module, device_ids, output_device, dim)
        self.wandb_input = dict()

    def get_wandb_input(self):
        net = self.module
        self.wandb_input = net.wandb_input
        return self.wandb_input

