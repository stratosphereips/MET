import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, \
    IterableDataset, random_split


def split_dataset(dataset, split_size):
    split_set_size = int(len(dataset) * split_size)
    rest_set_size = len(dataset) - split_set_size

    return random_split(dataset, [rest_set_size, split_set_size])


class MefDataset:
    def __init__(self,
                 batch_size,
                 train_set=None,
                 val_set=None,
                 test_set=None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size

    def generic_dataloader(self):
        dataset = []
        for set in [self.train_set, self.val_set, self.test_set]:
            if set is not None:
                dataset.append(set)

        if len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = ConcatDataset(dataset)

        return DataLoader(dataset=dataset, pin_memory=True,
                          num_workers=4, batch_size=self.batch_size)

    def train_dataloader(self):
        if isinstance(self.train_set, IterableDataset):
            return DataLoader(dataset=self.train_set, pin_memory=True,
                              shuffle=True, batch_size=self.batch_size)
        return DataLoader(dataset=self.train_set, pin_memory=True,
                          num_workers=4, shuffle=True,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        if isinstance(self.val_set, IterableDataset):
            return DataLoader(dataset=self.val_set, pin_memory=True,
                              batch_size=self.batch_size)
        return DataLoader(dataset=self.val_set, pin_memory=True,
                          num_workers=4, batch_size=self.batch_size)

    def test_dataloader(self):
        if isinstance(self.test_set, IterableDataset):
            return DataLoader(dataset=self.test_set, pin_memory=True,
                              batch_size=self.batch_size)
        return DataLoader(dataset=self.test_set, pin_memory=True,
                          num_workers=4, batch_size=self.batch_size)


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
        sample = self.data[index]
        target = self.targets[index]

        # When the target corresponds to integer return integer and not tensor
        if target.numel() == 1:
            target = target.item()

        return sample, target

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
    def __init__(self, victim_model, generator, latent_dim, batch_size,
                 output_type, greyscale="False"):
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
