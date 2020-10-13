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

    def __init__(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index][0], self.y[index]

    def __len__(self):
        return len(self.x)


class CustomDataset(Dataset):
    """
    Create completely new dataset from torch tensors or numpy arrays
    representing x, y
    """

    def __init__(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class NoYDataset(Dataset):
    """
    Dataset with only X
    """

    def __init__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


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
