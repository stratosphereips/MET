from pathlib import Path
from typing import Any

import numpy as np
import torch
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from torch.utils.data import Dataset, random_split


def split_dataset(dataset: Dataset, split_size: float):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size

    return random_split(dataset, [rest_set_size, split_set_size])


class CustomLabelDataset(Dataset):
    """
    Dataset that uses existing dataset with custom labels
    """

    def __init__(self, dataset: Dataset, targets: torch.Tensor):
        self.dataset = dataset
        self.targets = targets
        super().__init__()

    def __getitem__(self, index):
        return self.dataset[index][0], self.targets[index]

    def __len__(self):
        return len(self.dataset)


class NoYDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, index):
        # Returning random dummy integer as target because pytorch requires
        # that batch contains only tensor or numpy array and not NoneType
        return self.data[index], torch.randint(1, (1,))

    def __len__(self):
        return len(self.data)


class SavedDataset(Dataset):
    def __init__(self, save_loc: Path, targets: torch.Tensor, cuda_device: int = None):
        self._save_loc = save_loc
        self.targets = targets
        self.cuda_device = cuda_device

    def __getitem__(self, index):
        image = torch.load(
            self._save_loc.joinpath(f"{index}.pt"),
        )
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.targets)


class NumpyDataset(SklearnDataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_transform: Any = None,
        y_transform: Any = None,
    ) -> None:
        super().__init__(X, y, X_transform, y_transform)
