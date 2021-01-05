from pathlib import Path

from torchvision.datasets import STL10


class Stl10(STL10):
    def __init__(self,
                 root,
                 split="train",
                 folds=None,
                 transform=None,
                 target_transform=None,
                 download=True):
        root = Path(root).joinpath("stl10")
        super().__init__(root, split, folds, transform, target_transform,
                         download)
