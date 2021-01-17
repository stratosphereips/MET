from typing import Any, Tuple, Type

import torch
from torch.utils.data import Dataset, random_split


def split_dataset(dataset: Type[Dataset],
                  split_size: float):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size

    return random_split(dataset, [rest_set_size, split_set_size])


class CustomLabelDataset(Dataset):
    """
    Dataset that uses existing dataset with custom labels
    """

    def __init__(self,
                 dataset: Type[Dataset],
                 targets: torch.Tensor):
        self.dataset = dataset
        self.targets = targets
        super().__init__()

    def __getitem__(self, index):
        return self.dataset[index][0], self.targets[index]

    def __len__(self):
        return len(self.dataset)


class NoYDataset(Dataset):
    def __init__(self,
                 data: torch.Tensor):
        self.data = data

    def __getitem__(self, index):
        # Returning random dummy integer as target because pytorch requires
        # that batch contains only tensor or numpy array and not NoneType
        return self.data[index], torch.randint(1, (1,))

    def __len__(self):
        return len(self.data)
