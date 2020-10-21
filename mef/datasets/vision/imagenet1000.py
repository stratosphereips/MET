from pathlib import Path

from torchvision.datasets import ImageFolder


class ImageNet1000(ImageFolder):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, seed=None):
        root = Path(root)
        if train and "train" not in str(root):
            root = root.joinpath("train")
        elif not train and "val" not in str(root):
            root = root.joinpath("val")

        if not root.exists():
            raise ValueError("Imagenet2012 dataset not found at {}"
                             .format(str(root)))

        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self._seed = seed

        print("Loaded {} ({}) with {} samples".format(self.__class__.__name__,
                                                      "train" if train else
                                                      "validation",
                                                      len(self.samples)))
