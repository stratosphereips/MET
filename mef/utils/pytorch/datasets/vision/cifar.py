from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import CIFAR10, CIFAR100


class Cifar10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        root = Path(root).joinpath("cifar10")
        super().__init__(root, train, transform, target_transform, download)


class Cifar100(CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        root = Path(root).joinpath("cifar100")
        super().__init__(root, train, transform, target_transform, download)
