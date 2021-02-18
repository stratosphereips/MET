# Based on https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff
# /datasets/indoor67.py

from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import ImageFolder


class Indoor67(ImageFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        root = Path(root)
        if "Indoor67" not in str(root):
            root = Path(root).joinpath("Indoor67")

        if not root.exists():
            raise ValueError(
                "Dataset not found at {}. Please download it from {"
                "}.".format(root, "http://web.mit.edu/torralba/www/indoor.html")
            )

        # Initialize ImageFolder
        super().__init__(
            root=Path(root).joinpath("Images"),
            transform=transform,
            target_transform=target_transform,
        )
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs["train" if train else "test"]

        # Prune (self.imgs, self.samples to only include examples from the
        # required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print(
            "Loaded {} ({}) with {} examples".format(
                self.__class__.__name__, "train" if train else "test", len(self.samples)
            )
        )

    def get_partition_to_idxs(self):
        partition_to_idxs = {"train": [], "test": []}

        # ----------------- Load list of test images
        test_images = set()
        with open(self.root.joinpath("TestImages.txt")) as f:
            for line in f:
                test_images.add(line.strip())

        for idx, (filepath, _) in enumerate(self.samples):
            filepath = Path(filepath)
            filepath = str(filepath.relative_to(filepath.parent.parent)).replace(
                "\\", "/"
            )
            if filepath in test_images:
                partition_to_idxs["test"].append(idx)
            else:
                partition_to_idxs["train"].append(idx)

        return partition_to_idxs


if __name__ == "__main__":
    indoor = Indoor67("E:/Datasets/Indoor67")
