import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, random_split


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


class GeneratorRandomDataset(IterableDataset):
    def __init__(self, victim_model, generator, latent_dim, batch_size=64,
                 output_type="softmax", greyscale="False"):
        self._victim_model = victim_model
        self._generator = generator
        self._latent_dim = latent_dim
        self._output_type = output_type
        self._greyscale = greyscale
        self._batch_size = batch_size

    def __iter__(self):
        for _ in range(1000):
            images = self._generator(torch.Tensor(
                    np.random.uniform(-3.3, 3.3, size=(self._batch_size,
                                                       self._encoding_size))))

            if self._greyscale:
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers)
                images = images * multipliers
                images = images.sum(axis=1, keepdims=True)

            with torch.no_grad():
                y_preds = self._victim_model(images)
                if self._output_type == "one_hot":
                    labels = F.one_hot(torch.argmax(y_preds, dim=-1),
                                       num_classes=y_preds.size()[1])
                    # to_oneshot returns tensor with uint8 type
                    labels = labels.float()
                elif self._output_type == "softmax":
                    labels = F.softmax(y_preds, dim=-1)
                elif self._output_type == "labels":
                    labels = torch.argmax(y_preds, dim=-1)
                else:
                    labels = y_preds
            yield images, labels
