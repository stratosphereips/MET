import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.activethief import ActiveThief
from mef.models.vision.simplenet import SimpleNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_data
from mef.utils.pytorch.lighting.training import train_victim_model


def parse_args():
    parser = argparse.ArgumentParser("ActiveThief - MNIST example")
    parser.add_argument("-c", "--config_file", type=str,
                        default="./config.yaml",
                        help="path to configuration file")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="path to folder where datasets will be "
                             "downloaded")
    parser.add_argument("-s", "--save_loc", type=str,
                        default="./cache/activethief/MNIST",
                        help="path to folder where attack's files will be "
                             "saved")
    parser.add_argument("-t", "--train_epochs", type=int, default=10,
                        help="number of training epochs for the victim model")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="number of gpus to be used for training of "
                             "victim model")

    args = parser.parse_args()
    return args


def set_up(args):
    data_dir = args.data_dir
    save_loc = args.save_loc
    victim_save_loc = save_loc + "/victim/"
    training_epochs = args.train_epochs
    gpus = args.gpus
    dims = (1, 28, 28)

    victim_model = SimpleNet(input_dimensions=dims, num_classes=10)
    substitute_model = SimpleNet(input_dimensions=dims, num_classes=10)

    if gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = [transforms.Resize(dims[1:]), transforms.ToTensor()]
    mnist = dict()
    mnist["train"] = MNIST(root=data_dir, download=True,
                           transform=transforms.Compose(transform))

    mnist["test"] = MNIST(root=data_dir, train=False, download=True,
                          transform=transforms.Compose(transform))

    transform = [transforms.Resize(dims[1:]), transforms.Grayscale(),
                 transforms.ToTensor()]
    cifar10 = dict()
    cifar10["train"] = CIFAR10(root=data_dir, download=True,
                               transform=transforms.Compose(transform))
    cifar10["test"] = CIFAR10(root=data_dir, download=True, train=False,
                              transform=transforms.Compose(transform))

    cifar10["all"] = ConcatDataset([cifar10["train"], cifar10["test"]])

    test_set = mnist["test"]
    thief_dataset = mnist["train"]  # cifar10["all"]
    # Train secret model
    try:
        saved_model = torch.load(victim_save_loc + "final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        train_set, val_set = split_data(mnist["train"], 0.2)
        train_victim_model(victim_model, optimizer, loss, train_set, val_set,
                           training_epochs, victim_save_loc, gpus)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   victim_save_loc + "final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                test_set=test_set, thief_dataset=thief_dataset, num_classes=10)


if __name__ == "__main__":
    args = parse_args()
    mef.Test(args.config_file)
    mkdir_if_missing(args.save_loc)

    attack_variables = set_up(args)
    ActiveThief(**attack_variables, save_loc=args.save_loc).run()
