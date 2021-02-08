from collections import Callable
from pathlib import Path
from typing import Optional

from torchvision.datasets import FashionMNIST, MNIST


class Mnist(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        root = Path(root).joinpath("mnist")
        super().__init__(root, train, transform, target_transform, download)


class FashionMnist(FashionMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        root = Path(root).joinpath("fashion_mnist")
        super().__init__(root, train, transform, target_transform, download)
