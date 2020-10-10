import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from mef.attacks.blackbox import BlackBox
from mef.models.vision.simplenet import SimpleNet

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset, split_dataset
from mef.utils.pytorch.lighting.training import train_victim_model

SEED = 0
DATA_DIR = "./data"
SAVE_LOC = "./cache/blackbox/MNIST"
VICT_TRAIN_EPOCHS = 10
HOLDOUT = 150  # number of samples from MNIST-test to holdout for the attack
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

    train_loader = DataLoader(mnist["train"], batch_size=len(mnist["train"]))
    test_loader = DataLoader(mnist["test"], batch_size=len(mnist["test"]))

    x_test = next(iter(test_loader))[0].numpy()
    y_test = next(iter(test_loader))[1].numpy()

    x_sub = x_test[:HOLDOUT]
    y_sub = y_test[:HOLDOUT]

    x_test = x_test[HOLDOUT:]
    y_test = y_test[HOLDOUT:]

    x_train = next(iter(train_loader))[0].numpy()
    y_train = next(iter(train_loader))[1].numpy()

    # Train secret model
    try:
        saved_model = torch.load(SAVE_LOC + "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        # Prepare secret model
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
                x_test=x_test, y_test=y_test, num_classes=10), x_sub, y_sub


if __name__ == "__main__":
    mef.Test(gpus=GPUS, seed=SEED)
    mkdir_if_missing(SAVE_LOC)

    attack_variables, x_sub, y_sub = set_up()
    BlackBox(**attack_variables, save_loc=SAVE_LOC).run(x_sub, y_sub)
