import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import CIFAR10, ImageNet, STL10
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset, split_dataset
from mef.utils.pytorch.lighting.training import train_victim_model

SEED = 0
DATA_DIR = "./data"
IMAGENET_DIR = "E:\Datasets\ImageNet2012"
SAVE_LOC = "./cache/copycat/GOC"
NPD_SIZE = 2000  # Number of images taken from (non-problem) ImageNet dataset
VICT_TRAIN_EPOCHS = 10
GPUS = 1
DIMS = (3, 64, 64)


class GOCData:

    def __init__(self, npd_size, imagenet_dir, data_dir="./data",
                 dims=(3, 64, 64)):
        super().__init__()
        self.npd_size = npd_size
        self.imagenet_dir = imagenet_dir
        self.data_dir = data_dir

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transforms_list = [transforms.Resize(dims[1:]), transforms.ToTensor(),
                           transforms.Normalize(mean, std)]
        self.transform1 = transforms.Compose(transforms_list)
        self.transform2 = transforms.Compose(transforms_list[:-1])

        self.dims = dims

        self.test_set = None
        self.od_dataset = None
        self.thief_dataset = None

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        STL10(self.data_dir, download=True)
        STL10(self.data_dir, split="test", download=True)

    def _remove_class(self, class_to_remove, labels, data, class_to_idx,
                      classes):
        class_idx = class_to_idx[class_to_remove]

        class_idxes = np.where(labels == np.atleast_1d(class_idx))
        labels = np.delete(labels, class_idxes)
        labels[labels > class_idx] -= 1

        data = np.delete(data, class_idxes, 0)

        classes.remove(class_to_remove)

        # class_to_idx update
        class_to_idx.pop(class_to_remove)
        for name, idx in class_to_idx.items():
            if idx > class_idx:
                class_to_idx[name] -= 1

        return labels.tolist(), data, class_to_idx, classes

    def setup(self):
        cifar10 = dict()
        cifar10["train"] = CIFAR10(self.data_dir, transform=self.transform1)
        cifar10["test"] = CIFAR10(self.data_dir, train=False,
                                  transform=self.transform1)

        stl10 = dict()
        stl10["train"] = STL10(self.data_dir, transform=self.transform1)
        stl10["test"] = STL10(self.data_dir, split="test",
                              transform=self.transform1)

        # Replace car with automobile to make the class name same as in the
        # cifar10
        for setx in stl10.values():
            for i, cls in enumerate(setx.classes):
                if cls == "car":
                    setx.classes[i] = "automobile"
                    setx.classes.sort()
                    break

        # Remove frog class from CIFAR-10 and monkey from STL-10 so both
        # datasets have same class
        for name, setx in cifar10.items():
            setx.targets, setx.data, setx.class_to_idx, setx.classes = \
                self._remove_class("frog", setx.targets, setx.data,
                                   setx.class_to_idx, setx.classes)

        for name, setx in stl10.items():
            stl10_class_to_idx = {cls: idx for cls, idx in
                                  zip(setx.classes, range(len(
                                          setx.classes)))}
            setx.labels, setx.data, stl10_class_to_idx, setx.classes = \
                self._remove_class("monkey", setx.labels, setx.data,
                                   stl10_class_to_idx,
                                   setx.classes)

        imagenet = dict()
        imagenet["train"] = ImageNet(self.imagenet_dir,
                                     transform=self.transform2)
        imagenet["test"] = ImageNet(self.imagenet_dir, split="val",
                                    transform=self.transform2)
        imagenet["all"] = ConcatDataset(imagenet.values())

        self.test_set = cifar10["test"]
        self.od_dataset = cifar10["train"]
        pd_dataset = ConcatDataset([stl10["train"], stl10["test"]])
        npd_dataset, _ = random_split(imagenet["all"],
                                      [self.npd_size,
                                       len(imagenet["all"]) - self.npd_size])
        self.thief_dataset = ConcatDataset([pd_dataset, npd_dataset])


def set_up():
    victim_model = Vgg(vgg_type="vgg_16", input_dimensions=DIMS, num_classes=9)
    substitute_model = Vgg(vgg_type="vgg_16", input_dimensions=DIMS,
                           num_classes=9)

    if GPUS:
        victim_model.cuda()
        substitute_model.cuda()

    print("Preparing data")
    goc = GOCData(NPD_SIZE, IMAGENET_DIR, data_dir=DATA_DIR, dims=DIMS)
    goc.prepare_data()
    goc.setup()

    test_loader = DataLoader(goc.test_set, batch_size=len(goc.test_set))

    x_test = next(iter(test_loader))[0].numpy()
    y_test = next(iter(test_loader))[1].numpy()

    od_loader = DataLoader(goc.od_dataset, batch_size=len(goc.od_dataset))

    x_od = next(iter(od_loader))[0].numpy()
    y_od = next(iter(od_loader))[1].numpy()

    thief_loader = DataLoader(goc.thief_dataset,
                              batch_size=len(goc.thief_dataset))

    x_thief = next(iter(thief_loader))[0].numpy()
    y_thief = next(iter(thief_loader))[1].numpy()

    # Prepare target model
    try:
        saved_model = torch.load(SAVE_LOC + "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.01,
                                    momentum=0.8)
        loss = F.cross_entropy

        data = CustomDataset(x_od, y_od)
        train_set, val_set = split_dataset(data, 0.2)
        train_victim_model(victim_model, optimizer, loss, train_set, val_set,
                           VICT_TRAIN_EPOCHS, SAVE_LOC + "/victim/", GPUS)
        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                x_test=x_test, y_test=y_test), x_thief, y_thief


if __name__ == "__main__":
    mef.Test(gpus=GPUS, seed=SEED)
    mkdir_if_missing(SAVE_LOC)

    attack_variables, x_thief, y_thief = set_up()
    copycat = CopyCat(**attack_variables, save_loc=SAVE_LOC)
    copycat.run(x_thief, y_thief)
