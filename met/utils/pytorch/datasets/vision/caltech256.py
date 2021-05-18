# Based on https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff
# /datasets/caltech256.py

from collections import defaultdict as dd
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from torchvision.datasets import ImageFolder


class Caltech256(ImageFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 0,
    ):
        root = Path(root)
        if "256_ObjectCategories" not in str(root):
            root = Path(root).joinpath("256_ObjectCategories")

        if not root.exists():
            raise ValueError("Caltech256 dataset was not found in {}".format(str(root)))

        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )

        self._delete_clutter_class()
        self._seed = seed

        # Recommended value from Caltech256 dataset paper
        self._test_size = 25
        self._partition_to_idxs = self._get_partition_to_idxs()
        self._pruned_idxs = self._partition_to_idxs["train" if train else "test"]

        # Prune (self.imgs, self.samples to only include examples_paper from the
        # required train/test partition
        self.targets = [self.targets[i] for i in self._pruned_idxs]
        self.samples = [self.samples[i] for i in self._pruned_idxs]
        self.imgs = self.samples

        print(
            "Loaded {} ({}) with {} samples".format(
                self.__class__.__name__, "train" if train else "test", len(self.samples)
            )
        )

    def _delete_clutter_class(self) -> None:
        idx_clutter = self.class_to_idx["257.clutter"]
        self.samples = [s for s in self.samples if s[1] != idx_clutter]
        self.class_to_idx.pop("257.clutter")
        self.classes = self.classes[:-1]

    def _get_partition_to_idxs(self) -> Dict[str, List[int]]:
        partition_to_idxs = {"train": [], "test": []}

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
            # A constant no. kept aside for evaluation
            partition_to_idxs["test"] += idxs[: self._test_size]
            # Train on remaining
            partition_to_idxs["train"] += idxs[self._test_size :]

        np.random.set_state(before_state)

        return partition_to_idxs
