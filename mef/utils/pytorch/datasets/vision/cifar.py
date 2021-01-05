from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100


class Cifar10(CIFAR10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        root = Path(root).joinpath("cifar10")
        super().__init__(root, train, transform, target_transform, download)


class Cifar100(CIFAR100):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        root = Path(root).joinpath("cifar100")
        super().__init__(root, train, transform, target_transform, download)
