import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.atlas import AtlasThief
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets.vision import ImageNet1000
from mef.utils.pytorch.models.vision import AtCnn
from mef.utils.pytorch.datasets import split_dataset

IMAGENET_TRAIN_SIZE = 100000
IMAGENET_VAL_SIZE = 50000
DIMS = (3, 32, 32)
NUM_CLASSES = 10


def set_up(args):
    seed_everything(args.seed)

    victim_model = AtCnn(dims=DIMS, num_classes=NUM_CLASSES,
                         dropout_keep_prob=0.2)
    substitute_model = AtCnn(dims=DIMS, num_classes=NUM_CLASSES,
                             dropout_keep_prob=0.2)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    mean = (0.5,)
    std = (0.5,)
    transform = T.Compose([T.CenterCrop(DIMS[1]), T.ToTensor(),
                           T.Normalize(mean, std)])
    train_set = CIFAR10(root=args.cifar10_dir, download=True,
                        transform=transform)
    test_set = CIFAR10(root=args.cifar10_dir, train=False, download=True,
                       transform=transform)

    transform = T.Compose([T.Resize(DIMS[-1]), T.ToTensor(),
                           T.Normalize(mean, std)])
    imagenet_train = ImageNet1000(root=args.imagenet_dir,
                                  size=IMAGENET_TRAIN_SIZE,
                                  transform=transform, seed=args.seed)
    imagenet_val = ImageNet1000(root=args.imagenet_dir, train=False,
                                size=IMAGENET_VAL_SIZE, transform=transform,
                                seed=args.seed)
    thief_dataset = ConcatDataset([imagenet_train, imagenet_val])

    train_set, val_set = split_dataset(train_set, 0.2)
    optimizer = torch.optim.Adam(victim_model.parameters(),
                                 weight_decay=1e-3)
    loss = F.cross_entropy

    victim_training_epochs = 1000
    train_victim_model(victim_model, optimizer, loss, train_set,
                       NUM_CLASSES, victim_training_epochs, args.batch_size,
                       val_set, patience=args.patience,
                       save_loc=args.save_loc, gpus=args.gpus,
                       deterministic=args.deterministic, debug=args.debug,
                       precision=args.precision)

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    parser = AtlasThief.get_attack_args()
    parser.add_argument("--cifar10_dir", default="./data/", type=str,
                        help="Path to CIFAR10 dataset (Default: ./data/)")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, thief_dataset, test_set = set_up(args)
    af = AtlasThief(victim_model, substitute_model, NUM_CLASSES,
                    args.iterations, args.victim_output_type, args.budget)

    # Baset settings
    af.base_settings.save_loc = Path(args.save_loc)
    af.base_settings.gpus = args.gpus
    af.base_settings.num_workes = args.num_workers
    af.base_settings.batch_size = args.batch_size
    af.base_settings.seed = args.seed
    af.base_settings.deterministic = args.deterministic
    af.base_settings.debug = args.debug

    # Trainer settings
    af.trainer_settings.training_epochs = args.training_epochs
    af.trainer_settings.patience = args.patience
    af.trainer_settings.evaluation_frequency = args.evaluation_frequency
    af.trainer_settings.precision = args.precision
    af.trainer_settings.accuracy = args.accuracy

    af(thief_dataset, test_set)
