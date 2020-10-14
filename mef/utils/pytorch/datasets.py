import torch
from torch.utils.data import Dataset, random_split


def split_dataset(dataset, split_size):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size

    return random_split(dataset, [rest_set_size, split_set_size])


class ListDataset(Dataset):
    def __init__(self, list_object):
        self.list_object = list_object

    def __getitem__(self, index):
        return self.list_object[index]

    def __len__(self):
        return len(self.list_object)


class CustomLabelDataset(Dataset):
    """
    Dataset that uses existing dataset with custom labels
    """

    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets

    def __getitem__(self, index):
        return self.dataset[index][0], self.targets[index]

    def __len__(self):
        return len(self.dataset)


class CustomDataset(Dataset):
    """
    Create completely new dataset from torch tensors or numpy arrays
    representing x, y
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


class NoYDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Returning random dummy integer as target because pytorch requires
        # that batch contains only tensor or numpy array and not NoneType
        return self.data[index], torch.randint(1, (1,))

    def __len__(self):
        return len(self.data)


class AugmentationDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[(index % len(self.data))]
        # imgaug works with numpy arrays
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy().transpose(1, 2, 0)
        sample = self.transform(sample)

        return sample, self.labels[index]

    def __len__(self):
        return len(self.data)
