import argparse
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

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.training import get_trainer

SEED = 0
SAVE_LOC = "./cache/blackbox/MNIST"
VICT_TRAIN_EPOCHS = 10
DIMS = (1, 28, 28)


def parse_args():
    parser = argparse.ArgumentParser(description="BlackBox model extraction "
                                                 "attack - Mnist example")
    parser.add_argument("-m", "--mnist_dir", default="./data", type=str,
                        help="Path to MNIST dataset")
    parser.add_argument("-h", "--holdout", default=150, type=int,
                        help="Number of samples from MNIST-test to holdout "
                             "for the attack")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="Number of gpus to be used")
    args = parser.parse_args()

    return args


def set_up(args):
    victim_model = SimpleNet(input_dimensions=DIMS, num_classes=10)
    substitute_model = SimpleNet(input_dimensions=DIMS, num_classes=10)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = transforms.Compose([transforms.CenterCrop(DIMS[1:]),
                                    transforms.ToTensor()])
    mnist = dict()
    mnist["train"] = MNIST(root=args.mnist_dir, download=True,
                           transform=transform)
    mnist["test"] = MNIST(root=args.mnist_dir, train=False, download=True,
                          transform=transform)

    idx_test = np.random.permutation(len(mnist["test"]))
    idx_sub = idx_test[:args.holdout]
    idx_test = idx_test[args.holdout:]

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
        trainer = get_trainer(args.gpus, VICT_TRAIN_EPOCHS,
                              early_stop_tolerance=10,
                              save_loc=SAVE_LOC + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                num_classes=10), sub_dataset, test_set


if __name__ == "__main__":
    args = parse_args()

    mkdir_if_missing(SAVE_LOC)
    attack_variables, sub_dataset, test_set = set_up(args)

    bb = BlackBox(**attack_variables, save_loc=SAVE_LOC, gpus=args.gpus,
                  seed=SEED)
    bb.run(sub_dataset, test_set)
