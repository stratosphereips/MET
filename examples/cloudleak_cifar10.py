import argparse
import os
import sys

from mef.utils.pytorch.datasets import split_data
from mef.utils.pytorch.lighting.training import train_victim_model

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageNet, CIFAR10
from torchvision.transforms import transforms

import mef
from mef.utils.ios import mkdir_if_missing
from mef.attacks.cloudleak import CloudLeak
from mef.models.vision.vgg import Vgg


def parse_args():
    parser = argparse.ArgumentParser("CloudLeak - CIFAR10 example")
    parser.add_argument("-c", "--config_file", type=str,
                        default="./config.yaml",
                        help="path to configuration file (default: "
                             "./config.yaml)")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="path to folder where datasets will be "
                             "downloaded (default: ./data)")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="path to directory where ImageNet2012 is located")
    parser.add_argument("-f", "--thief_size", type=int, default=1000,
                        help="size of the thief dataset (default: 1000)")
    parser.add_argument("-s", "--save_loc", type=str,
                        default="./cache/cloudleak/CIFAR10",
                        help="path to folder where attack's files will be "
                             "saved (default: ./cache/cloudleak/CIFAR10)")
    parser.add_argument("-t", "--train_epochs", type=int, default=10,
                        help="number of training epochs for the secret model "
                             "(default: 10)")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="number of gpus to be used for training of "
                             "victim model")

    args = parser.parse_args()
    return args


def set_up(args):
    data_dir = args.data_dir
    imagenet_dir = args.imagenet_dir
    save_loc = args.save_loc
    victim_save_loc = save_loc + "/victim/"
    training_epochs = args.train_epochs
    gpus = args.gpus
    dims = (3, 224, 224)

    victim_model = Vgg(vgg_type="vgg_16", input_dimensions=dims,
                       num_classes=10)
    substitute_model = Vgg(vgg_type="vgg_16", input_dimensions=dims,
                           num_classes=10)

    if gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose(
            [transforms.Resize(dims[1:]), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    cifar10 = dict()
    cifar10["train"] = CIFAR10(root=data_dir, download=True,
                               transform=transform)
    cifar10["test"] = CIFAR10(root=data_dir, train=False, download=True,
                              transform=transform)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
            [transforms.Resize(dims[1:]), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    # imagenet = dict()
    # imagenet["train"] = ImageNet(imagenet_dir, transform=transform)
    # imagenet["test"] = ImageNet(imagenet_dir, split="val", transform=transform)
    # imagenet["all"] = ConcatDataset(imagenet.values())

    test_set = cifar10["test"]
    # thief_dataset, _ = random_split(imagenet["all"], [args.thief_size,
    # len(imagenet["all"]) -
    #                                                   args.thief_size])
    thief_dataset = cifar10["train"]

    # Train secret model
    try:
        saved_model = torch.load(victim_save_loc + "final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded target model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training secret model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        train_set, val_set = split_data(cifar10["train"], 0.2)
        train_victim_model(victim_model, optimizer, loss, train_set,
                           val_set, training_epochs, victim_save_loc, gpus)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   victim_save_loc + "final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                test_set=test_set, thief_dataset=thief_dataset, num_classes=10)


if __name__ == "__main__":
    args = parse_args()
    mef.Test(args.config_file)
    mkdir_if_missing(args.save_loc)

    variables = set_up(args)
    CloudLeak(**variables, save_loc=args.save_loc).run()
