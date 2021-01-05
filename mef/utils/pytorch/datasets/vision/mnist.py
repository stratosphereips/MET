from pathlib import Path

from torchvision.datasets import MNIST, FashionMNIST


class Mnist(MNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        root = Path(root).joinpath("mnist")
        super().__init__(root, train, transform, target_transform, download)


class FashionMnist(FashionMNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        root = Path(root).joinpath("fashion_mnist")
        super().__init__(root, train, transform, target_transform, download)