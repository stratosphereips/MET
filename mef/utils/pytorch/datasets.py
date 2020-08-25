from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, list_object):
        self.list_object = list_object

    def __getitem__(self, index):
        return self.list_object[index]

    def __len__(self):
        return len(self.list_object)


class CustomLabelDataset(Dataset):
    def __init__(self, dataset, labels):
        self.data = dataset
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index][0], self.labels[index]

    def __len__(self):
        return len(self.data)