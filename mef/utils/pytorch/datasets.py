import torch
from torch.utils.data import Dataset, random_split


def get_split_sizes(dataset, split_size):
    if isinstance(split_size, float):
        split_set_size = int(len(dataset) * split_size)
    elif isinstance(split_size, int):
        split_set_size = split_size
    else:
        raise ValueError("split_size must be either float or integer!")

    rest_set_size = len(dataset) - split_set_size

    return rest_set_size, split_set_size


def split_data(dataset, split_size):
    rest_set_size, split_set_size = get_split_sizes(dataset, split_size)

    rest_set, split_set = random_split(dataset, [rest_set_size,
                                                 split_set_size])

    return rest_set, split_set


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
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        return self.dataset[index][0], self.labels[index]

    def __len__(self):
        return len(self.dataset)

class CustomDataset(Dataset):
    """
    Create completely new dataset from data
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

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
