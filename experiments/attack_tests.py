import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import torch
import torchvision.transforms as T
import numpy as np
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset, Subset

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks import ActiveThief, BlackBox, CopyCat, KnockOff, Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision import Caltech256, ImageNet1000, Indoor67, Stl10
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import ResNet

IMAGENET_TRAIN_SIZE = 100000
IMAGENET_VAL_SIZE = 20000
TRAINING_EPOCHS = 200
PATIENCE = 50
BATCH_SIZE = 64
EVALUATION_FREQUENCY = 1
SEED = 200916

Dataset = namedtuple("Dataset", ["name", "class_", "num_classes"])
DATASETS = (
    # Dataset("STL10", Stl10, 10),
    Dataset("Indoor67", Indoor67, 67),
    Dataset("Caltech256", Caltech256, 256),
)
BUDGET = 20000
OUTPUT_TYPES = ("softmax", "one_hot")

AttackInfo = namedtuple("AttackInfo", ["name", "type"])
ME_ATTACKS = (
    AttackInfo("active-thief", "entropy"),
    AttackInfo("active-thief", "k-center"),
    # AttackInfo("active-thief", "dfal"),
    # AttackInfo("active-thief", "dfal+k-center"),
    AttackInfo("blackbox", ""),
    AttackInfo("copycat", ""),
    AttackInfo("knockoff", "nets_adaptive-cert"),
    AttackInfo("knockoff", "nets_adaptive-div"),
    AttackInfo("knockoff", "nets_adaptive-loss"),
    AttackInfo("knockoff", "nets_adaptive-all"),
)
ATTACKS_DICT = {
    "active-thief": ActiveThief,
    "blackbox": BlackBox,
    "copycat": CopyCat,
    "knockoff-nets": KnockOff,
    "ripper": Ripper,
}
ATTACKS_CONFIG = {
    "active-thief": {"iterations": 10, "save_samples": True},
    "blackbox": {"iterations": 6},
    "copycat": {},
    "knockoff-nets": {"save_samples": True},
    "ripper": {},
}


def getr_args():
    parser = ArgumentParser(description="Model extraction attacks STL10 test")
    parser.add_argument(
        "--stl10_dir",
        default="./cache/data",
        type=str,
        help="Location where STL10 dataset is or should be "
        "downloaded to (Default: ./cache/data)",
    )
    parser.add_argument(
        "--indoor67_dir",
        default="./cache/data",
        type=str,
        help="Path to Indoor67 dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--caltech256_dir",
        default="./cache/data",
        type=str,
        help="Path to Caltech256 dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--imagenet_dir",
        default="./cache/data",
        type=str,
        help="Path to ImageNet dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--save_loc",
        type=str,
        default="./cache/",
        help="Path where the attacks file should be " "saved (Default: " "./cache/)",
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus to be used (Default: 0)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used in loaders (" "Default: 1)",
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Precision of caluclation in bits must be "
        "one of {16, 32} (Default: 32)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode (Default: False)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getr_args()

    save_loc = Path(args.save_loc)

    imagenet_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    imagenet_train = ImageNet1000(
        root=args.imagenet_dir,
        size=IMAGENET_TRAIN_SIZE,
        transform=imagenet_transform,
        seed=SEED,
    )
    imagenet_val = ImageNet1000(
        root=args.imagenet_dir,
        train=False,
        size=IMAGENET_VAL_SIZE,
        transform=imagenet_transform,
        seed=SEED,
    )

    for dataset in DATASETS:
        seed_everything(SEED)
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        train_kwargs = {"transform": train_transform}
        test_kwargs = {"transform": test_transform, "train": False}
        if dataset.name == "STL10":
            dataset_dir = args.stl10_dir
        elif dataset.name == "Indoor67":
            dataset_dir = args.indoor67_dir
        else:
            train_kwargs["seed"] = SEED
            test_kwargs["seed"] = SEED
            dataset_dir = args.caltech256_dir

        train_set = dataset.class_(dataset_dir, **train_kwargs)
        train_set, val_set = split_dataset(train_set, 0.2)
        test_set = dataset.class_(dataset_dir, **test_kwargs)

        # Prepare victim model
        victim_model = ResNet("resnet_34", num_classes=dataset.num_classes)
        victim_optimizer = torch.optim.SGD(
            victim_model.parameters(), lr=0.1, momentum=0.5
        )
        victim_loss = torch.nn.functional.cross_entropy
        lr_scheduler = torch.optim.lr_scheduler.StepLR(victim_optimizer, step_size=60)
        train_victim_model(
            victim_model,
            victim_optimizer,
            victim_loss,
            train_set,
            dataset.num_classes,
            TRAINING_EPOCHS,
            BATCH_SIZE,
            args.num_workers,
            val_set=val_set,
            test_set=test_set,
            lr_scheduler=lr_scheduler,
            gpus=args.gpus,
            save_loc=save_loc.joinpath("Attack-tests", dataset.name),
            debug=args.debug,
        )

        for attack in ME_ATTACKS:
            for output_type in OUTPUT_TYPES:
                attack_save_loc = save_loc.joinpath(
                    "Attack-tests",
                    dataset.name,
                    f"budget-{BUDGET}",
                    attack.name,
                    attack.type,
                )

                # Check if the attack is already done
                final_substitute_model = attack_save_loc.joinpath(
                    "substitute", "final_substitute_model-state_dict.pt"
                )
                if final_substitute_model.exists():
                    continue

                # Prepare models for the attack
                kwargs = {
                    "victim_model": VictimModel(
                        victim_model, dataset.num_classes, output_type
                    )
                }
                # Prepare substitute model
                substitute_model = ResNet("resnet_34", num_classes=dataset.num_classes,)
                substitute_optimizer = torch.optim.SGD(
                    substitute_model.parameters(), lr=0.01, momentum=0.5
                )
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    substitute_optimizer, step_size=60
                )
                substitute_loss = soft_cross_entropy
                kwargs["substitute_model"] = TrainableModel(
                    substitute_model,
                    dataset.num_classes,
                    substitute_optimizer,
                    substitute_loss,
                    lr_scheduler,
                )

                kwargs.update(ATTACKS_CONFIG[attack.name])

                # Add attack specific key-name arguments
                if attack.name == "active-thief":
                    kwargs["selection_strategy"] = attack.type
                    kwargs["budget"] = BUDGET
                elif attack.name == "knockoff":
                    sampling_strategy, reward_type = attack.type.split("-")
                    kwargs["sampling_strategy"] = sampling_strategy
                    kwargs["reward_type"] = reward_type
                    kwargs["budget"] = BUDGET
                elif attack.name == "blackbox":
                    kwargs["budget"] = BUDGET

                attack_instance = ATTACKS_DICT[attack.name](**kwargs)

                # Base settings
                attack_instance.base_settings.save_loc = attack_save_loc
                attack_instance.base_settings.gpus = args.gpus
                attack_instance.base_settings.num_workers = args.num_workers
                attack_instance.base_settings.batch_size = BATCH_SIZE
                attack_instance.base_settings.seed = SEED
                attack_instance.base_settings.deterministic = True
                attack_instance.base_settings.debug = args.debug

                # Trainer settings
                attack_instance.trainer_settings.training_epochs = TRAINING_EPOCHS
                attack_instance.trainer_settings.patience = PATIENCE
                attack_instance.trainer_settings.evaluation_frequency = (
                    EVALUATION_FREQUENCY
                )
                attack_instance.trainer_settings.precision = args.precision
                attack_instance.trainer_settings.use_accuracy = False

                run_kwargs = {
                    "sub_data": ConcatDataset([imagenet_train, imagenet_val]),
                    "test_set": test_set,
                }

                if "copycat":
                    idxs_all = np.arange(len(run_kwargs["sub_data"]))
                    idxs_sub = np.random.permutation(idxs_all)[:BUDGET]
                    run_kwargs["sub_data"] = Subset(run_kwargs["sub_data"], idxs_sub)

                attack_instance(**run_kwargs)
