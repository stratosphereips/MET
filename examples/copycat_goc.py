import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageNet
from torchvision.transforms import transforms

from mef.utils.pytorch.datasets import split_data
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.ios import mkdir_if_missing


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


def parse_args():
    parser = argparse.ArgumentParser("ActiveThief - MNIST example")
    parser.add_argument("-c", "--config_file", type=str,
                        default="./config.yaml",
                        help="path to configuration file")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="path to directory where datasets will be "
                             "downloaded")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="path to directory where ImageNet2012 is located")
    parser.add_argument("-s", "--save_loc", type=str,
                        default="./cache/copycat/GOC",
                        help="path to folder where attack's files will be "
                             "saved")
    parser.add_argument("-n", "--npd_size", type=int, default=2000,
                        help="size of the non problem domain dataset that "
                             "should be taken from "
                             "ImageNet2012")
    parser.add_argument("-t", "--train_epochs", type=int, default=1000,
                        help="number of trainining epochs for the target "
                             "model and pd_ol model")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="number of gpus to be used for training of "
                             "victim model")

    args = parser.parse_args()
    return args


def train_model(model, dataset, training_epochs, save_loc, gpus):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    loss = F.cross_entropy
    mef_model = MefModule(model, optimizer, loss)

    trainer = get_trainer(gpus, training_epochs, save_loc=save_loc)

    train_set, val_set = split_data(dataset, split_size=0.2)

    train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                  pin_memory=True,
                                  num_workers=4, batch_size=64)
    val_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                pin_memory=True,
                                num_workers=4, batch_size=64)

    trainer.fit(mef_model, train_dataloader, val_dataloader)

    return


def set_up(args):
    data_dir = args.data_dir
    imagenet_dir = args.imagenet_dir
    save_loc = args.save_loc
    victim_save_loc = save_loc + "/victim/"
    npd_size = args.npd_size
    training_epochs = args.train_epochs
    gpus = args.gpus
    dims = (3, 64, 64)

    victim_model = Vgg(vgg_type="vgg_16", input_dimensions=dims, num_classes=9)
    substitute_model = Vgg(vgg_type="vgg_16", input_dimensions=dims,
                           num_classes=9)

    if gpus:
        victim_model.cuda()
        substitute_model.cuda()

    print("Preparing data")
    goc = GOCData(npd_size, imagenet_dir, data_dir=data_dir, dims=dims)
    goc.prepare_data()
    goc.setup()

    # Prepare target model
    try:
        saved_model = torch.load(victim_save_loc + "final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded target model")
    except FileNotFoundError:
        print("Training victim model")
        train_model(victim_model, goc.od_dataset, training_epochs,
                    victim_save_loc, gpus)
        torch.save(dict(state_dict=victim_model.state_dict()),
                   victim_save_loc +
                   "final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                test_set=goc.test_set, thief_dataset=goc.thief_dataset)


if __name__ == "__main__":
    args = parse_args()
    mef.Test(args.config_file)
    mkdir_if_missing(args.save_loc)

    attack_variables = set_up(args)
    copycat = CopyCat(**attack_variables, save_loc=args.save_loc).run()
