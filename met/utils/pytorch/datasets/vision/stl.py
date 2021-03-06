from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import STL10


class Stl10(STL10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        root = Path(root).joinpath("stl10")
        super().__init__(
            root,
            "train" if train else "test",
            folds,
            transform,
            target_transform,
            download,
        )
        self.targets = self.labels
