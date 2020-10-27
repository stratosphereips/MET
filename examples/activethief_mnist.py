import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.activethief import ActiveThief
from mef.datasets.vision.imagenet1000 import ImageNet1000
from mef.models.vision.simplenet import SimpleNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

SEED = 0
SAVE_LOC = "./cache/activethief/MNIST"
VICT_TRAIN_EPOCHS = 10
DIMS = (1, 28, 28)


def parse_args():
    parser = argparse.ArgumentParser(description="Activethief model "
                                                 "extraction attack - Mnist "
                                                 "example")
    parser.add_argument("-s", "--selection_strategy", default="entropy",
                        type=str, help="Activethief selection strategy can "
                                       "be one of {random, entropy, "
                                       "k-center, dfal, dfal+k-center}")
    parser.add_argument("-m", "--mnist_dir", default="./data", type=str,
                        help="Path to MNIST dataset")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
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
    transform = transforms.Compose([transforms.CenterCrop(DIMS[2]),
                                    transforms.ToTensor()])
    mnist = dict()
    mnist["train"] = MNIST(root=args.mnist_dir, download=True,
                           transform=transform)
    mnist["test"] = MNIST(root=args.mnist_dir, train=False, download=True,
                          transform=transform)
    test_set = mnist["test"]
    train_set = mnist["train"]

    transform = transforms.Compose([transforms.CenterCrop(DIMS[2]),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    imagenet = dict()
    imagenet["train"] = ImageNet1000(root=args.imagenet_dir,
                                     transform=transform)
    imagenet["val"] = ImageNet1000(root=args.imagenet_dir, train=False,
                                   transform=transform)
    imagenet["all"] = ConcatDataset(imagenet.values())
    sub_dataset = imagenet["all"]

    try:
        saved_model = torch.load(SAVE_LOC + "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        train_set, val_set = split_dataset(train_set, 0.2)
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

    models = dict(victim_model=victim_model, substitute_model=substitute_model)
    return models, sub_dataset, test_set


if __name__ == "__main__":
    args = parse_args()

    mkdir_if_missing(SAVE_LOC)
    attack_variables, sub_dataset, test_set = set_up(args)

    af = ActiveThief(**attack_variables, num_classes=10,
                     selection_strategy=args.selection_strategy,
                     save_loc=SAVE_LOC, gpus=args.gpus, seed=SEED)
    af.run(sub_dataset, test_set)
