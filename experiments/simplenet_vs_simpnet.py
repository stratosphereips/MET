import os
import sys
from argparse import ArgumentParser

import torch
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from met.utils.experiment import train_victim_model
from met.utils.pytorch.datasets.vision import Cifar10
from met.utils.pytorch.models.vision import SimpleNet, SimpNet

NUM_CLASSES = 10
EPOCHS = 1000
PATIENCE = 100
EVALUATION_FREQUENCY = 1
BATCH_SIZE = 100


def getr_args():
    parser = ArgumentParser(description="Simplenet experiment")
    parser.add_argument(
        "--cifar10_dir",
        default="./cache/data",
        type=str,
        help="Location where Cifar10 dataset is or should be "
        "downloaded to (Default: ./cache/data)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="Number of gpu to be used (Default: 0)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used in loaders (" "Default: 1)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getr_args()

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )
    test_transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    train_set = Cifar10(args.cifar10_dir, download=True, transform=train_transform)
    test_set = Cifar10(
        args.cifar10_dir, train=False, download=True, transform=test_transform
    )

    simplenet = SimpleNet(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(simplenet.parameters())
    loss = torch.nn.functional.cross_entropy
    train_victim_model(
        simplenet,
        optimizer,
        loss,
        train_set,
        NUM_CLASSES,
        EPOCHS,
        BATCH_SIZE,
        args.num_workers,
        val_set=test_set,
        test_set=test_set,
        patience=PATIENCE,
        gpu=args.gpu,
        evaluation_frequency=EVALUATION_FREQUENCY,
        save_loc="./cache/Simplenet-vs-Simpnet-cifar10/SimpleNet/victim",
    )

    simpnet1 = SimpNet(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(simpnet1.parameters())
    loss = torch.nn.functional.cross_entropy
    train_victim_model(
        simpnet1,
        optimizer,
        loss,
        train_set,
        NUM_CLASSES,
        EPOCHS,
        BATCH_SIZE,
        args.num_workers,
        val_set=test_set,
        test_set=test_set,
        patience=PATIENCE,
        gpu=args.gpu,
        evaluation_frequency=EVALUATION_FREQUENCY,
        save_loc="./cache/Simplenet-vs-Simpnet-cifar10" "/SimpNet-5M",
    )

    simpnet2 = SimpNet(num_classes=NUM_CLASSES, less_parameters=False)
    optimizer = torch.optim.Adam(simpnet2.parameters())
    loss = torch.nn.functional.cross_entropy
    train_victim_model(
        simpnet2,
        optimizer,
        loss,
        train_set,
        NUM_CLASSES,
        EPOCHS,
        BATCH_SIZE,
        args.num_workers,
        val_set=test_set,
        test_set=test_set,
        patience=PATIENCE,
        gpu=args.gpu,
        evaluation_frequency=EVALUATION_FREQUENCY,
        save_loc="./cache/Simplenet-vs-Simpnet-cifar10" "/SimpNet-8M",
    )
