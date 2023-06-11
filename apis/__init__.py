# from apis.cifar.train import train, train2
from .tester import Tester
from .trainer import Trainer
from .aug_trainer import AugTrainer, AugTester

__all__ = [
    'Trainer', 'Tester', 'AugTrainer', 'AugTester',
]