import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, \
    IterableDataset, random_split


def split_dataset(dataset, split_size):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size

    return random_split(dataset, [rest_set_size, split_set_size])


class MefDataset:
    def __init__(self,
                 base_settings,
                 train_set=None,
                 val_set=None,
                 test_set=None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self._base_settings = base_settings

    def generic_dataloader(self):
        dataset = []
        for set in [self.train_set, self.val_set, self.test_set]:
            if set is not None:
                dataset.append(set)

        if len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = ConcatDataset(dataset)

        return DataLoader(dataset=dataset,
                          pin_memory=self._base_settings.gpus != 0,
                          num_workers=self._base_settings.num_workers,
                          batch_size=self._base_settings.batch_size)

    def train_dataloader(self):
        if isinstance(self.train_set, IterableDataset):
            return DataLoader(dataset=self.train_set)
        return DataLoader(dataset=self.train_set,
                          pin_memory=self._base_settings.gpus != 0,
                          num_workers=self._base_settings.num_workers,
                          shuffle=True,
                          batch_size=self._base_settings.batch_size)

    def val_dataloader(self):
        if isinstance(self.val_set, IterableDataset):
            return DataLoader(dataset=self.val_set)
        return DataLoader(dataset=self.val_set,
                          pin_memory=self._base_settings.gpus != 0,
                          num_workers=self._base_settings.num_workers,
                          batch_size=self._base_settings.batch_size)

    def test_dataloader(self):
        if isinstance(self.test_set, IterableDataset):
            return DataLoader(dataset=self.test_set)
        return DataLoader(dataset=self.test_set,
                          pin_memory=self._base_settings.gpus != 0,
                          num_workers=self._base_settings.num_workers,
                          batch_size=self._base_settings.batch_size)


class CustomLabelDataset(Dataset):
    """
    Dataset that uses existing dataset with custom labels
    """

    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets
        super().__init__()

    def __getitem__(self, index):
        return self.dataset[index][0], self.targets[index]

    def __len__(self):
        return len(self.dataset)

class TensorDadaset(Dataset):
    """
    Create completely new dataset from torch tensors representing x, y
    """

    def __init__(self, data, targets):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)

        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

class NumpyDataset(TensorDadaset):
    """
    Create completely new dataset from torch numpy arrays representing x, y
    """
    def __init__(self, data, targets):
        data = torch.from_numpy(data)
        targets = torch.from_numpy(targets)
        super().__init__(data, targets)


class NoYDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Returning random dummy integer as target because pytorch requires
        # that batch contains only tensor or numpy array and not NoneType
        return self.data[index], torch.randint(1, (1,))

    def __len__(self):
        return len(self.data)
