import os
import sys
from argparse import ArgumentParser

import torch
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets.vision import Cifar10
from mef.utils.pytorch.models.vision import SimpNet, SimpleNet

NUM_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 64


def getr_args():
    parser = ArgumentParser(description="Simplenet experiment")
    parser.add_argument("--cifar10_dir", default="./cache/data", type=str,
                        help="Location where Cifar10 dataset is or should be "
                             "downloaded to (Default: ./cache/data)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of gpus to be used (Default: 0)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to be used in loaders ("
                             "Default: 1)")

    return parser.parse_args()


if __name__ == "__main__":
    args = getr_args()

    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize((0.5,), (0.5,))])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize((0.5,), (0.5,))])

    train_set = Cifar10(args.cifar10_dir, download=True,
                        transform=train_transform)
    test_set = Cifar10(args.cifar10_dir, train=False, download=True,
                       transform=test_transform)

    simplenet = SimpleNet(num_classes=NUM_CLASSES)
    optimizer = torch.optim.SGD(simplenet.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.005)
    loss = torch.nn.functional.cross_entropy
    train_victim_model(simplenet, optimizer, loss, train_set, NUM_CLASSES,
                       EPOCHS, BATCH_SIZE, args.num_workers, test_set=test_set,
                       gpus=args.gpus,
                       save_loc="./cache/Simplenet-vs-Simpnet-cifar10"
                                "/SimpleNet")

    simpnet1 = SimpNet(num_classes=NUM_CLASSES)
    optimizer = torch.optim.SGD(simpnet1.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.005)
    loss = torch.nn.functional.cross_entropy
    train_victim_model(simpnet1, optimizer, loss, train_set, NUM_CLASSES,
                       EPOCHS, BATCH_SIZE, args.num_workers, test_set=test_set,
                       gpus=args.gpus,
                       save_loc="./cache/Simplenet-vs-Simpnet-cifar10"
                                "/SimpNet-5M")

    simpnet2 = SimpNet(num_classes=NUM_CLASSES, less_parameters=False)
    optimizer = torch.optim.SGD(simpnet2.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.005)
    loss = torch.nn.functional.cross_entropy
    train_victim_model(simpnet2, optimizer, loss, train_set, NUM_CLASSES,
                       EPOCHS, BATCH_SIZE, args.num_workers, test_set=test_set,
                       gpus=args.gpus,
                       save_loc="./cache/Simplenet-vs-Simpnet-cifar10"
                                "/SimpNet-8M")
