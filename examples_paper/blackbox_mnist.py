import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.blackbox import BlackBox
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.blocks import ConvBlock, MaxPoolLayer
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision import Mnist
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import GenericCNN

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 10
DIMS = (1, 28, 28)
BOUNDS = (-1, 1)


def set_up(args):
    seed_everything(args.seed)

    victim_model = GenericCNN(
        dims=DIMS,
        num_classes=NUM_CLASSES,
        conv_blocks=(
            ConvBlock(1, 32, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            MaxPoolLayer(2, 2),
        ),
        fc_layers=(200,),
    )
    substitute_model = GenericCNN(
        dims=DIMS,
        num_classes=NUM_CLASSES,
        conv_blocks=(
            ConvBlock(1, 32, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            MaxPoolLayer(2, 2),
        ),
        fc_layers=(200,),
    )

    # Prepare data
    print("Preparing data")
    transform = T.Compose(
        [T.Resize(DIMS[-1]), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
    )
    train_set = Mnist(root=args.mnist_dir, transform=transform, download=True)
    test_set = Mnist(
        root=args.mnist_dir, train=False, transform=transform, download=True
    )

    train_set, pre_attacker_set = split_dataset(train_set, 0.1)

    # Get from each class same number of samples
    loader = DataLoader(
        dataset=pre_attacker_set,
        pin_memory=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    targets = []
    for _, labels in loader:
        targets.append(labels)
    targets = torch.cat(targets)

    targets = np.array(targets)
    idx_adversary = []
    for cls in range(NUM_CLASSES):
        idx_cls = np.where(targets == cls)[0]
        idx_selected = np.random.permutation(idx_cls)[:SAMPLES_PER_CLASS]

        idx_adversary.extend(idx_selected)

    idx_test = np.random.permutation(len(pre_attacker_set))
    idx_test = np.setdiff1d(idx_test, idx_adversary)

    thief_dataset = Subset(pre_attacker_set, idx_adversary)

    train_victim_model(
        victim_model,
        torch.optim.Adam,
        F.cross_entropy,
        train_set,
        NUM_CLASSES,
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_set=test_set,
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpu=args.gpu,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )

    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="labels")
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.Adam,
        F.cross_entropy,
    )

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    parser = BlackBox.get_attack_args()
    parser.add_argument(
        "--mnist_dir",
        default="./data/",
        type=str,
        help="Path to MNIST dataset (Default: ./data/)",
    )
    args = parser.parse_args()
    args.training_epochs = 100
    args.iterations = 10
    args.lmbda = 64 / 255

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, thief_dataset, test_set = set_up(args)
    bb = BlackBox(victim_model, substitute_model, BOUNDS, args.iterations, args.lmbda)

    # Baset settings
    bb.base_settings.save_loc = Path(args.save_loc)
    bb.base_settings.gpu = args.gpu
    bb.base_settings.num_workers = args.num_workers
    bb.base_settings.batch_size = args.batch_size
    bb.base_settings.seed = args.seed
    bb.base_settings.deterministic = args.deterministic
    bb.base_settings.debug = args.debug

    # Trainer settings
    bb.trainer_settings.training_epochs = args.training_epochs
    bb.trainer_settings.precision = args.precision
    bb.trainer_settings.use_accuracy = args.accuracy

    bb(thief_dataset, test_set)
