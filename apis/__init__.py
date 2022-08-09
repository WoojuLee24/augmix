# from apis.cifar.train import train, train2
from apis.cifar.train import Trainer
from apis.cifar.test import test, test_c, test_c_dg

__all__ = [
    #'train', 'train2',
    'Trainer',
    'test', 'test_c', 'test_c_dg'
]