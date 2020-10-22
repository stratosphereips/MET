import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import CIFAR10, STL10
from torchvision.transforms import transforms

from mef.datasets.vision.imagenet1000 import ImageNet1000
from mef.utils.pytorch.lighting.module import MefModule

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.training import get_trainer

SEED = 0
SAVE_LOC = "./cache/copycat/GOC"
VICT_TRAIN_EPOCHS = 10


class GOCData:

    def __init__(self, npd_size, imagenet_dir, stl10_dir, cifar10_dir):
        super().__init__()
        self.npd_size = npd_size
        self.imagenet_dir = imagenet_dir
        self.stl10_dir = stl10_dir
        self.cifar10_dir = cifar10_dir
        self.dims = (3, 224, 224)

        # Imagenet values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms_list = [transforms.CenterCrop(self.dims[2]),
                           transforms.ToTensor(),
                           transforms.Normalize(mean, std)]
        self.transform = transforms.Compose(transforms_list)

        self.test_set = None
        self.od_dataset = None
        self.sub_dataset = None

    def prepare_data(self):
        # download
        CIFAR10(self.cifar10_dir, download=True)
        CIFAR10(self.cifar10_dir, train=False, download=True)
        STL10(self.stl10_dir, download=True)
        STL10(self.stl10_dir, split="test", download=True)

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
        cifar10["train"] = CIFAR10(self.cifar10_dir, transform=self.transform)
        cifar10["test"] = CIFAR10(self.cifar10_dir, train=False,
                                  transform=self.transform)

        stl10 = dict()
        stl10["train"] = STL10(self.stl10_dir, transform=self.transform)
        stl10["test"] = STL10(self.stl10_dir, split="test",
                              transform=self.transform)

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
        imagenet["train"] = ImageNet1000(self.imagenet_dir,
                                         transform=self.transform)
        imagenet["val"] = ImageNet1000(self.imagenet_dir, train=False,
                                       transform=self.transform)
        imagenet["all"] = ConcatDataset(imagenet.values())

        self.test_set = cifar10["test"]
        self.od_dataset = cifar10["train"]
        pd_dataset = ConcatDataset([stl10["train"], stl10["test"]])
        npd_dataset, _ = random_split(imagenet["all"],
                                      [self.npd_size,
                                       len(imagenet["all"]) - self.npd_size])
        self.sub_dataset = ConcatDataset([pd_dataset, npd_dataset])


def parse_args():
    parser = argparse.ArgumentParser(description="CopyCat model extraction "
                                                 "attack - General Object "
                                                 "Classification (GOC) "
                                                 "example")
    parser.add_argument("-s", "--stl10_dir", default="./data", type=str,
                        help="Path to Stl10 dataset")
    parser.add_argument("-c", "--cifar10_dir", default="./data", type=str,
                        help="Path to Cifar10 dataset")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    parser.add_argument("-n", "--npd_size", type=int, default=120000,
                        help="Number of images taken from (non-problem) "
                             "ImageNet dataset")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="Number of gpus to be used")

    args = parser.parse_args()

    return args


def set_up(args):
    victim_model = Vgg(vgg_type="vgg_16", num_classes=9)
    substitute_model = Vgg(vgg_type="vgg_16", num_classes=9)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    print("Preparing data")
    goc = GOCData(args.npd_size, args.imagenet_dir,
                  cifar10_dir=args.cifar10_dir, stl10_dir=args.stl10_dir)
    goc.prepare_data()
    goc.setup()

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

        train_set, val_set = split_dataset(goc.od_dataset, 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=32)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=32)

        mef_model = MefModule(victim_model, optimizer, loss)
        trainer = get_trainer(args.gpus, VICT_TRAIN_EPOCHS,
                              early_stop_tolerance=10,
                              save_loc=SAVE_LOC + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model,
                substitute_model=substitute_model), goc.sub_dataset, \
           goc.test_set


if __name__ == "__main__":
    args = parse_args()

    mkdir_if_missing(SAVE_LOC)
    attack_variables, sub_dataset, test_set = set_up(args)

    copycat = CopyCat(**attack_variables, save_loc=SAVE_LOC, gpus=args.gpus,
                      seed=SEED)
    copycat.run(sub_dataset, test_set)
