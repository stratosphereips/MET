import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from mef.attacks.blackbox import BlackBox
from mef.models.vision.simplenet import SimpleNet
from mef.utils.pytorch.lighting.module import MefModule

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.training import get_trainer

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

    idx_test = np.random.permutation(len(mnist["test"]))

    idx_sub = idx_test[:HOLDOUT]
    idx_test = idx_test[HOLDOUT:]

    sub_dataset = Subset(mnist["test"], idx_sub)
    test_set = Subset(mnist["test"], idx_test)

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

        train_set, val_set = split_dataset(mnist["train"], 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=64)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=64)

        mef_model = MefModule(victim_model, optimizer, loss)
        trainer = get_trainer(GPUS, VICT_TRAIN_EPOCHS, early_stop_tolerance=10,
                              save_loc=SAVE_LOC + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                num_classes=10), sub_dataset, test_set


if __name__ == "__main__":
    mef.Test(gpus=GPUS, seed=SEED)
    mkdir_if_missing(SAVE_LOC)

    attack_variables, sub_dataset, test_set = set_up()
    BlackBox(**attack_variables, save_loc=SAVE_LOC).run(sub_dataset, test_set)
