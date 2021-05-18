import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from met.attacks.knockoff import KnockOff
from met.utils.experiment import train_victim_model
from met.utils.ios import mkdir_if_missing
from met.utils.pytorch.datasets.vision import Caltech256, ImageNet1000
from met.utils.pytorch.functional import soft_cross_entropy
from met.utils.pytorch.lightning.module import TrainableModel, VictimModel
from met.utils.pytorch.models.vision import ResNet

NUM_CLASSES = 256
DIMS = (3, 224, 224)
IMAGENET_NORMALIZATION = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def set_up(args):
    seed_everything(args.seed)

    victim_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)

    # Prepare data
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            IMAGENET_NORMALIZATION,
        ]
    )
    test_transform = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), IMAGENET_NORMALIZATION]
    )

    train_set = Caltech256(
        args.caltech256_dir, transform=train_transform, seed=args.seed
    )
    test_set = Caltech256(
        args.caltech256_dir, train=False, transform=test_transform, seed=args.seed
    )
    imagenet = ImageNet1000(args.imagenet_dir, transform=test_transform, seed=args.seed)
    caltech_train_no_aug = Caltech256(
        args.caltech256_dir,
        transform=test_transform,
        seed=args.seed,
    )

    vict_training_epochs = 200
    train_victim_model(
        victim_model,
        torch.optim.SGD,
        F.cross_entropy,
        train_set,
        NUM_CLASSES,
        vict_training_epochs,
        args.batch_size,
        args.num_workers,
        optimizer_args={"lr": 0.1, "momentum": 0.5},
        test_set=test_set,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_args={"step_size": 60},
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
        torch.optim.SGD,
        soft_cross_entropy,
        torch.optim.lr_scheduler.StepLR,
        optimizer_args={"lr": 0.01, "momentum": 0.5},
        lr_scheduler_args={"step_size": 60},
    )

    # Because we are using adaptive_flat we are using the same experiment
    # setup as in the paper, where it is assumed that the attacker has access
    # to all available data
    sub_dataset = ConcatDataset([imagenet, caltech_train_no_aug])
    sub_dataset.num_classes = 1256
    sub_dataset.datasets[1].targets = [
        y + 1000 for y in sub_dataset.datasets[1].targets
    ]
    sub_dataset.targets = []
    sub_dataset.targets.extend(sub_dataset.datasets[0].targets)
    sub_dataset.targets.extend(sub_dataset.datasets[1].targets)

    return victim_model, substitute_model, sub_dataset, test_set


if __name__ == "__main__":
    parser = KnockOff.get_attack_args()
    parser.add_argument(
        "--caltech256_dir",
        default="./data/",
        type=str,
        help="Path to Caltech256 dataset (Default: ./data/",
    )
    parser.add_argument("--imagenet_dir", type=str, help="Path to ImageNet dataset")
    args = parser.parse_args()
    args.training_epochs = 100

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, sub_dataset, test_set = set_up(args)
    ko = KnockOff(
        victim_model,
        substitute_model,
        torch.optim.SGD(substitute_model.parameters(), lr=0.0005, momentum=0.5),
        args.sampling_strategy,
        args.reward_type,
        args.budget,
    )

    # Baset settings
    ko.base_settings.save_loc = Path(args.save_loc)
    ko.base_settings.gpu = args.gpu
    ko.base_settings.num_workers = args.num_workers
    ko.base_settings.batch_size = args.batch_size
    ko.base_settings.seed = args.seed
    ko.base_settings.deterministic = args.deterministic
    ko.base_settings.debug = args.debug

    # Trainer settings
    ko.trainer_settings.training_epochs = args.training_epochs
    ko.trainer_settings.precision = args.precision
    ko.trainer_settings.use_accuracy = args.accuracy

    ko(sub_dataset, test_set)
