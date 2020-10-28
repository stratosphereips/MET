from collections import defaultdict as dd
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder


class ImageNet1000(ImageFolder):

    def __init__(self, root, size=None, train=True, transform=None,
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

        self._size = size
        self._seed = seed

        if size is not None:
            self._subset_idxs = self._get_subset_idxs()
            self.samples = [self.samples[i] for i in self._subset_idxs]
            self.imgs = self.samples

        print("Loaded {} ({}) with {} samples".format(self.__class__.__name__,
                                                      "train" if train else
                                                      "validation",
                                                      len(self.samples)))

    def _get_subset_idxs(self):
        samples_per_class = self._size // 1000
        partition_to_idxs = []

        # Use this random seed to make partition consistent
        before_state = np.random.get_state()
        np.random.seed(self._seed)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs += idxs[:samples_per_class]

        np.random.set_state(before_state)

        return partition_to_idxs
