import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.attacks.activethief import ActiveThief
from mef.models.vision.simplenet import SimpleNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset, split_dataset
from mef.utils.pytorch.lighting.training import train_victim_model

SELECTION_STRATEGY = "dfal"
SEED = 0
DATA_DIR = "./data"
SAVE_LOC = "./cache/activethief/MNIST"
VICT_TRAIN_EPOCHS = 10
GPUS = 1
DIMS = (1, 28, 28)


def set_up():
    victim_model = SimpleNet(input_dimensions=DIMS, num_classes=10)
    substitute_model = SimpleNet(input_dimensions=DIMS, num_classes=10)

    if GPUS:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = [transforms.Resize(DIMS[1:]), transforms.ToTensor()]
    mnist = dict()
    mnist["train"] = MNIST(root=DATA_DIR, download=True,
                           transform=transforms.Compose(transform))

    mnist["test"] = MNIST(root=DATA_DIR, train=False, download=True,
                          transform=transforms.Compose(transform))

    test_loader = DataLoader(mnist["test"], batch_size=len(mnist["test"]))

    x_test = next(iter(test_loader))[0].numpy()
    y_test = next(iter(test_loader))[1].numpy()

    train_loader = DataLoader(mnist["train"], batch_size=len(mnist["train"]))

    x_train = next(iter(train_loader))[0].numpy()
    y_train = next(iter(train_loader))[1].numpy()

    transform = [transforms.Resize(DIMS[1:]), transforms.Grayscale(),
                 transforms.ToTensor()]
    cifar10 = dict()
    cifar10["train"] = CIFAR10(root=DATA_DIR, download=True,
                               transform=transforms.Compose(transform))
    cifar10["test"] = CIFAR10(root=DATA_DIR, download=True, train=False,
                              transform=transforms.Compose(transform))

    cifar10["all"] = ConcatDataset([cifar10["train"], cifar10["test"]])

    thief_loader = DataLoader(cifar10["all"], batch_size=len(cifar10["all"]))

    x_thief = next(iter(thief_loader))[0].numpy()
    y_thief = next(iter(thief_loader))[1].numpy()

    try:
        saved_model = torch.load(SAVE_LOC + "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        data = CustomDataset(x_train, y_train)
        train_set, val_set = split_dataset(data, 0.2)
        train_victim_model(victim_model, optimizer, loss, train_set, val_set,
                           VICT_TRAIN_EPOCHS, SAVE_LOC + "/victim/", GPUS)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                x_test=x_test, y_test=y_test, num_classes=10), x_thief, y_thief


if __name__ == "__main__":
    mef.Test(gpus=GPUS, seed=SEED)
    mkdir_if_missing(SAVE_LOC)

    attack_variables, x_sub, y_sub = set_up()
    af = ActiveThief(**attack_variables, selection_strategy=SELECTION_STRATEGY,
                     save_loc=SAVE_LOC)
    af.run(x_sub, y_sub)
