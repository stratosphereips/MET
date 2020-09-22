import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageNet
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.details import ModelDetails
from mef.utils.ios import mkdir_if_missing


def parse_args():
    parser = argparse.ArgumentParser("ActiveThief - MNIST example")
    parser.add_argument("-c", "--config_file", type=str, default="./config.yaml",
                        help="path to configuration file")
    parser.add_argument("-d", "--data_root", type=str, default="./data",
                        help="path to folder where datasets will be downloaded")
    parser.add_argument("-i", "--imagenet_root", type=str,
                        help="path to folder where ImageNet2012 is located")
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
    data_root = args.data_root
    imagenet_root = args.imagenet_root
    save_loc = args.save_loc
    target_save_loc = save_loc + "/final_target_model.pt"
    opd_save_loc = save_loc + "/final_opd_model.pt"
    npd_size = args.npd_size
    train_epochs = args.train_epochs
    device = "cuda" if args.gpu else "cpu"

    model_details = ModelDetails(net=dict(name="vgg_16",
                                          act="relu",
                                          drop="none",
                                          pool="max_2",
                                          ks=3,
                                          n_conv=13,
                                          n_fc=3))
    target_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                       model_details=model_details)
    opd_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                    model_details=model_details)
    copycat_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                        model_details=model_details)

    if device == "cuda":
        target_model.cuda()
        opd_model.cuda()
        copycat_model.cuda()

    print("Preparing data")
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean,
                                                                                          std)]

    cifar10 = dict()
    cifar10["train"] = CIFAR10(root=data_root, download=True,
                               transform=transforms.Compose(transform))

    cifar10["test"] = CIFAR10(root=data_root, train=False, download=True,
                              transform=transforms.Compose(transform))

    transform = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean,
                                                                                          std)]
    stl10 = dict()
    stl10["train"] = STL10(root=data_root, transform=transforms.Compose(transform),
                           download=True)
    stl10["test"] = STL10(root=data_root, split="test",
                          transform=transforms.Compose(transform),
                          download=True)

    # Replace car with automobile to make the class name same as in the cifar10
    for setx in stl10.values():
        for i, cls in enumerate(setx.classes):
            if cls == "car":
                setx.classes[i] = "automobile"
                setx.classes.sort()
                break

    transform = [transforms.Resize((64, 64)), transforms.ToTensor()]
    imagenet = dict()
    imagenet["train"] = ImageNet(root=imagenet_root, transform=transforms.Compose(transform))
    imagenet["test"] = ImageNet(root=imagenet_root, split="val",
                                transform=transforms.Compose(transform))
    imagenet["all"] = ConcatDataset(imagenet.values())
    idx = np.arange(len(imagenet["all"]))
    npd_idx = np.random.choice(idx, size=npd_size, replace=False)

    # Remove frog class from CIFAR-10 and monkey from STL-10 so both datasets have same class
    for name, setx in cifar10.items():
        setx.targets, setx.data, setx.class_to_idx, setx.classes = remove_class("frog",
                                                                                setx.targets,
                                                                                setx.data,
                                                                                setx.class_to_idx,
                                                                                setx.classes)
    for name, setx in stl10.items():
        stl10_class_to_idx = {cls: idx for cls, idx in zip(setx.classes, range(len(
            setx.classes)))}
        setx.labels, setx.data, stl10_class_to_idx, setx.classes = remove_class("monkey",
                                                                                setx.labels,
                                                                                setx.data,
                                                                                stl10_class_to_idx,
                                                                                setx.classes)

    test_dataset = cifar10["test"]
    od_dataset = cifar10["train"]
    pd_dataset = ConcatDataset([stl10["train"], stl10["test"]])
    npd_dataset = Subset(imagenet["all"], npd_idx)

    # Prepare target model
    try:
        saved_model = torch.load(target_save_loc)
        target_model.load_state_dict(saved_model["state_dict"])
        print("Loaded target model")
    except FileNotFoundError:
        print("Training target model")
        train_model(target_model, od_dataset, target_save_loc, train_epochs, device)

    # Prepare PD-OL model
    try:
        saved_model = torch.load(opd_save_loc)
        opd_model.load_state_dict(saved_model["state_dict"])
        print("Loaded PD-OL model")
    except FileNotFoundError:
        print("Training PD-OL model")
        train_model(opd_model, pd_dataset, target_save_loc, train_epochs, device)

    return dict(target_model=target_model, opd_model=opd_model, copycat_model=copycat_model,
                test_dataset=test_dataset, pd_dataset=pd_dataset, npd_dataset=npd_dataset)


if __name__ == "__main__":
    args = parse_args()
    mkdir_if_missing(args.save_loc)
    attack_variables = set_up(args)

    mef.Test(args.config_file)
    copycat = CopyCat(**attack_variables, save_loc=args.save_loc).run()
