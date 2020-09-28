import torch
from torch.utils.data import Dataset, random_split


def split_data(dataset, split_size):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size
    rest_set, split_set = random_split(dataset,
                                       [rest_set_size, split_set_size])

    return rest_set, split_set


class ListDataset(Dataset):
    def __init__(self, list_object):
        self.list_object = list_object

    def __getitem__(self, index):
        return self.list_object[index]

    def __len__(self):
        return len(self.list_object)


class CustomLabelDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        return self.dataset[index][0], self.labels[index]

    def __len__(self):
        return len(self.dataset)


class AugmentationDataset(Dataset):
    def __init__(self, data, labels, transform=None,
                 augmentation_multiplier=1):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augmentation_multiplier = augmentation_multiplier

    def __getitem__(self, index):
        if index < index % len(self.data):
            return self.data[index], self.labels[index]
        else:
            sample = self.data[(index % len(self.data))]
            if self.transform:
                if isinstance(sample, torch.Tensor):
                    sample = sample.numpy().transpose(1, 2, 0)
                sample = self.transform(sample)

        return sample, self.labels[(index % len(self.data))]

    def __len__(self):
        # Pytorch is performing transformations on the fly so in order to
        # simulate bigger dataset
        # we multiply the size by augmentation multiplier
        return len(self.data) * self.augmentation_multiplier
