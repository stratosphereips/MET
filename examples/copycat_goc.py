import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer
from torch.utils.data import ConcatDataset, Subset, DataLoader, random_split
from torchvision.datasets import CIFAR10, STL10, ImageNet
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.ios import mkdir_if_missing

class GOCData:

    def __init__(self, npd_size, imagenet_dir, data_dir="./data", dims=(3, 64, 64)):
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

        self.test_dataset = None
        self.od_dataset = None
        self.pd_dataset = None
        self.npd_dataset = None

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        STL10(self.data_dir, download=True)
        STL10(self.data_dir, split="test", download=True)

    def _remove_class(self, class_to_remove, labels, data, class_to_idx, classes):
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
        cifar10["test"] = CIFAR10(self.data_dir, train=False, transform=self.transform1)

        stl10 = dict()
        stl10["train"] = STL10(self.data_dir, transform=self.transform1)
        stl10["test"] = STL10(self.data_dir, split="test", transform=self.transform1)

        # Replace car with automobile to make the class name same as in the cifar10
        for setx in stl10.values():
            for i, cls in enumerate(setx.classes):
                if cls == "car":
                    setx.classes[i] = "automobile"
                    setx.classes.sort()
                    break

        # Remove frog class from CIFAR-10 and monkey from STL-10 so both datasets have same class
        for name, setx in cifar10.items():
            setx.targets, setx.data, setx.class_to_idx, setx.classes = \
                self._remove_class("frog", setx.targets, setx.data, setx.class_to_idx, setx.classes)

        for name, setx in stl10.items():
            stl10_class_to_idx = {cls: idx for cls, idx in zip(setx.classes, range(len(
                setx.classes)))}
            setx.labels, setx.data, stl10_class_to_idx, setx.classes = \
                self._remove_class("monkey", setx.labels, setx.data, stl10_class_to_idx,
                                   setx.classes)

        imagenet = dict()
        imagenet["train"] = ImageNet(self.imagenet_dir, transform=self.transform2)
        imagenet["test"] = ImageNet(self.imagenet_dir, split="val", transform=self.transform2)
        imagenet["all"] = ConcatDataset(imagenet.values())

        self.test_dataset = cifar10["test"]
        self.od_dataset = cifar10["train"]
        self.pd_dataset = ConcatDataset([stl10["train"], stl10["test"]])
        self.npd_dataset, _ = random_split(imagenet["all"], [self.npd_size, len(imagenet["all"]) -
                                                             self.npd_size])

def parse_args():
    parser = argparse.ArgumentParser("ActiveThief - MNIST example")
    parser.add_argument("-c", "--config_file", type=str, default="./config.yaml",
                        help="path to configuration file")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="path to directory where datasets will be downloaded")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="path to directory where ImageNet2012 is located")
    parser.add_argument("-s", "--save_loc", type=str, default="./cache/copycat/GOC",
                        help="path to folder where attack's files will be saved")
    parser.add_argument("-n", "--npd_size", type=int, default=2000,
                        help="size of the non problem domain dataset that should be taken from "
                             "ImageNet2012")
    parser.add_argument("-t", "--train_epochs", type=int, default=10,
                        help="number of trainining epochs for the target model and pd_ol model")
    parser.add_argument("-g", "--gpu", type=bool, default=False,
                        help="whether gpu should be used")

    args = parser.parse_args()
    return args


def remove_class(class_to_remove, labels, data, class_to_idx, classes):
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


def train_model(model, train_data, save_loc, train_epochs, device):
    train_loader = DataLoader(dataset=train_data, batch_size=128,
                              shuffle=True, num_workers=1, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    loss_function = F.cross_entropy
    trainer = create_supervised_trainer(model, optimizer, loss_function,
                                        device=device)

    ProgressBar().attach(trainer)
    trainer.run(train_loader, max_epochs=train_epochs)

    torch.save(dict(state_dict=model.state_dict()), save_loc)

    return


def set_up(args):
    data_dir = args.data_dir
    imagenet_dir = args.imagenet_dir
    save_loc = args.save_loc
    target_save_loc = save_loc + "/final_target_model.pt"
    opd_save_loc = save_loc + "/final_opd_model.pt"
    npd_size = args.npd_size
    train_epochs = args.train_epochs
    device = "cuda" if args.gpu else "cpu"
    dims = (3, 64, 64)

    target_model = Vgg(vgg_type="vgg_16", input_dimensions=dims, num_classes=9)
    opd_model = Vgg(vgg_type="vgg_16", input_dimensions=dims, num_classes=9)
    copycat_model = Vgg(vgg_type="vgg_16", input_dimensions=dims, num_classes=9)

    if device == "cuda":
        target_model.cuda()
        opd_model.cuda()
        copycat_model.cuda()

    print("Preparing data")
    goc = GOCData(npd_size, imagenet_dir, data_dir=data_dir, dims=dims)
    goc.prepare_data()
    goc.setup()

    # Prepare target model
    try:
        saved_model = torch.load(target_save_loc)
        target_model.load_state_dict(saved_model["state_dict"])
        print("Loaded target model")
    except FileNotFoundError:
        print("Training target model")
        train_model(target_model, goc.od_dataset, target_save_loc, train_epochs, device)

    # Prepare PD-OL model
    try:
        saved_model = torch.load(opd_save_loc)
        opd_model.load_state_dict(saved_model["state_dict"])
        print("Loaded PD-OL model")
    except FileNotFoundError:
        print("Training PD-OL model")
        train_model(opd_model, goc.pd_dataset, target_save_loc, train_epochs, device)

    return dict(target_model=target_model, opd_model=opd_model, copycat_model=copycat_model,
                test_dataset=goc.test_dataset, pd_dataset=goc.pd_dataset,
                npd_dataset=goc.npd_dataset)


if __name__ == "__main__":
    args = parse_args()
    mkdir_if_missing(args.save_loc)
    attack_variables = set_up(args)

    mef.Test(args.config_file)
    copycat = CopyCat(**attack_variables, num_classes=9, save_loc=args.save_loc).run()
