import argparse
import os
import sys

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.activethief import ActiveThief
from mef.models.vision.mnet import Mnet
from mef.utils.details import ModelDetails
from mef.utils.ios import mkdir_if_missing

input_dimension = (1, 28, 28)


def parse_args():
    parser = argparse.ArgumentParser("ActiveThief - MNIST example")
    parser.add_argument("-c", "--config_file", type=str, default="./config.yaml",
                        help="path to configuration file")
    parser.add_argument("-d", "--data_root", type=str, default="./data",
                        help="path to folder where datasets will be downloaded")
    parser.add_argument("-s", "--save_loc", type=str, default="./cache/activethief/MNIST",
                        help="path to folder where attack's files will be saved")
    parser.add_argument("-t", "--train_epochs", type=int, default=10,
                        help="number of trainining epochs for the secre_model")
    parser.add_argument("-g", "--gpu", type=bool, default=False,
                        help="whether gpu should be used")

    args = parser.parse_args()
    return args


def set_up(args):
    data_root = args.data_root
    save_loc = args.save_loc
    secret_savel_loc = save_loc + "/secret_model.pt"
    train_epochs = args.train_epochs
    device = "cuda" if args.gpu else "cpu"

    secret_details = ModelDetails(net=dict(name="mnet",
                                           act="relu",
                                           drop="none",
                                           pool="max_2",
                                           ks=3,
                                           n_conv=3,
                                           n_fc=3), )
    substitute_details = ModelDetails(net=dict(name="mnet",
                                               act="relu",
                                               drop="none",
                                               pool="max_2",
                                               ks=3,
                                               n_conv=3,
                                               n_fc=3))
    secret_model = Mnet(input_dimensions=input_dimension, num_classes=10,
                        model_details=secret_details)
    substitute_model = Mnet(input_dimensions=input_dimension, num_classes=10,
                            model_details=substitute_details)

    if device == "cuda":
        secret_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = [transforms.Resize(input_dimension[1:]), transforms.ToTensor()]
    mnist = dict()
    mnist["train"] = MNIST(root=data_root, download=True,
                           transform=transforms.Compose(transform))

    mnist["test"] = MNIST(root=data_root, train=False, download=True,
                          transform=transforms.Compose(transform))
    test_dataset = mnist["test"]

    transform = [transforms.Resize(input_dimension[1:]), transforms.Grayscale(),
                 transforms.ToTensor()]
    cifar10 = dict()
    cifar10["train"] = CIFAR10(root=data_root, download=True,
                               transform=transforms.Compose(transform))
    cifar10["test"] = CIFAR10(root=data_root, download=True,
                              train=False, transform=transforms.Compose(transform))

    thief_dataset = cifar10["train"]
    validation_dataset = cifar10["test"]

    # Train secret model
    try:
        saved_model = torch.load(secret_savel_loc)
        secret_model.load_state_dict(saved_model["state_dict"])
        print("Loaded target model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training secret model")
        train_loader = DataLoader(dataset=mnist["train"], batch_size=128, shuffle=True,
                                  num_workers=4, pin_memory=True)
        optimizer = torch.optim.Adam(secret_model.parameters())
        loss_function = F.cross_entropy
        trainer = create_supervised_trainer(secret_model, optimizer, loss_function,
                                            device=device)
        ProgressBar().attach(trainer)
        trainer.run(train_loader, max_epochs=train_epochs)

        torch.save(dict(state_dict=secret_model.state_dict()), secret_savel_loc)

    return dict(secret_model=secret_model, substitute_model=substitute_model,
                test_dataset=test_dataset,
                validation_dataset=validation_dataset, thief_dataset=thief_dataset)


if __name__ == "__main__":
    args = parse_args()
    mkdir_if_missing(args.save_loc)
    variables = set_up(args)

    mef.Test(args.config_file)
    ActiveThief(**variables, save_loc=args.save_loc).run()
