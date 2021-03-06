import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from met.attacks import ActiveThief
from met.utils.experiment import train_victim_model
from met.utils.ios import mkdir_if_missing
from met.utils.pytorch.datasets import split_dataset
from met.utils.pytorch.datasets.vision import Cifar10, ImageNet1000
from met.utils.pytorch.functional import soft_cross_entropy
from met.utils.pytorch.lightning.module import TrainableModel, VictimModel
from met.utils.pytorch.models.vision.GenericCNN import GenericCNN

IMAGENET_TRAIN_SIZE = 100000
IMAGENET_VAL_SIZE = 20000
DIMS = (3, 32, 32)
NUM_CLASSES = 10
BOUNDS = (0, 1)


def set_up(args):
    seed_everything(args.seed)

    victim_model = GenericCNN(dims=DIMS, num_classes=NUM_CLASSES, dropout_keep_prob=0.2)
    substitute_model = GenericCNN(
        dims=DIMS, num_classes=NUM_CLASSES, dropout_keep_prob=0.2
    )

    # Prepare data
    print("Preparing data")
    transform = T.Compose([T.ToTensor()])
    train_set = Cifar10(root=args.cifar10_dir, transform=transform)
    test_set = Cifar10(root=args.cifar10_dir, train=False, transform=transform)

    transform = T.Compose([T.Resize(DIMS[1:3]), T.ToTensor()])
    imagenet_train = ImageNet1000(
        root=args.imagenet_dir,
        size=IMAGENET_TRAIN_SIZE,
        transform=transform,
        seed=args.seed,
    )
    imagenet_val = ImageNet1000(
        root=args.imagenet_dir,
        train=False,
        size=IMAGENET_VAL_SIZE,
        transform=transform,
        seed=args.seed,
    )
    adversary_dataset = imagenet_train
    val_dataset = imagenet_val

    train_set, val_set = split_dataset(train_set, 0.2)

    train_victim_model(
        victim_model,
        torch.optim.Adam,
        F.cross_entropy,
        train_set,
        NUM_CLASSES,
        args.training_epochs,
        args.batch_size,
        args.num_workers,
        val_set=val_set,
        test_set=test_set,
        patience=args.patience,
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpu=args.gpu,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )

    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="softmax")
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.Adam,
        soft_cross_entropy,
    )

    return victim_model, substitute_model, adversary_dataset, test_set, val_dataset


if __name__ == "__main__":
    parser = ActiveThief.get_attack_args()
    parser.add_argument(
        "--cifar10_dir",
        default="./data/",
        type=str,
        help="Path to CIFAR10 dataset (Default: ./data/)",
    )
    parser.add_argument("--imagenet_dir", type=str, help="Path to ImageNet dataset")
    args = parser.parse_args()
    # Values from the ActiveThief paper
    args.training_epochs = 1000
    args.patience = 100
    args.evaluation_frequency = 1
    args.batch_size = 150

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, adversary_dataset, test_set, val_dataset = set_up(args)

    af = ActiveThief(
        victim_model,
        substitute_model,
        args.selection_strategy,
        args.iterations,
        args.budget,
        args.centers_per_iteration,
        args.deepfool_max_steps,
    )

    # Baset settings
    af.base_settings.save_loc = Path(args.save_loc)
    af.base_settings.gpu = args.gpu
    af.base_settings.num_workers = args.num_workers
    af.base_settings.batch_size = args.batch_size
    af.base_settings.seed = args.seed
    af.base_settings.deterministic = args.deterministic
    af.base_settings.debug = args.debug

    # Trainer settings
    af.trainer_settings.training_epochs = args.training_epochs
    af.trainer_settings.patience = args.patience
    af.trainer_settings.evaluation_frequency = args.evaluation_frequency
    af.trainer_settings.precision = args.precision
    af.trainer_settings.use_accuracy = args.accuracy

    af(adversary_dataset, test_set, val_dataset)
