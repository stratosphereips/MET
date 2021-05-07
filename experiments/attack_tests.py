import math
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset, ConcatDataset

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.base import AttackBase
from mef.attacks import ActiveThief, BlackBox, CopyCat, KnockOff, Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision import (
    GTSRB,
    Caltech256,
    Cifar10,
    Cifar100,
    FashionMnist,
    ImageNet1000,
    Stl10,
)
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import Generator, TrainableModel, VictimModel
from mef.utils.pytorch.models.generators import Sngan
from mef.utils.pytorch.models.vision import GenericCNN, ResNet, SimpleNet

VICT_TRAINING_EPOCHS = 200
SUB_TRAINING_EPOCHS = 100
BATCH_SIZE = 150
SEEDS = [200916, 211096]
BOUNDS = (-1, 1)

TestSettings = namedtuple(
    "TestSettings",
    [
        "name",
        "datasets",
        "substitute_dataset",
        "victim_output_types",
        "substitute_model_archs",
        "attacks_to_run",
    ],
)
AttackInfo = namedtuple("AttackInfo", ["name", "type", "budget"])
DatasetInfo = namedtuple(
    "DatasetInfo",
    [
        "name",
        "class_",
        "num_classes",
        "sample_dims",
        "train_transform",
        "test_transform",
        "substitute_dataset_transform",
    ],
)

DATASET_INFOS = {
    "FashionMNIST": DatasetInfo(
        "FashionMnist",
        FashionMnist,
        10,
        (1, 32, 32),
        T.Compose(
            [
                T.Pad(2),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.Pad(2), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose(
            [
                T.Resize((32, 32)),
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
    ),
    "Cifar10": DatasetInfo(
        "Cifar10",
        Cifar10,
        10,
        (3, 32, 32),
        T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
    ),
    "GTSRB": DatasetInfo(
        "GTSRB",
        GTSRB,
        43,
        (3, 32, 32),
        T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
    ),
    "GTSRB-128": DatasetInfo(
        "GTSRB",
        GTSRB,
        43,
        (3, 128, 128),
        T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
    ),
    "Caltech256": DatasetInfo(
        "Caltech256",
        Caltech256,
        256,
        (3, 128, 128),
        T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
    ),
}

TEST_SETTINGS = (
    TestSettings(
        "comparison_of_attacks_on_most_popular_datasets_from_paper",
        datasets=["FashionMNIST", "Cifar10", "GTSRB"],
        substitute_dataset=["ImageNet"],
        victim_output_types=["softmax"],
        substitute_model_archs=["simplenet"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("copycat", "", "All"),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
            AttackInfo("blackbox-ripper", "random", 20000),
            AttackInfo("blackbox-ripper", "optimized", 20000),
            AttackInfo("blackbox-ripper", "random", 120000),
            AttackInfo("blackbox-ripper", "optimized", 120000),
        ],
    ),
    TestSettings(
        "influence_in_terms_of_victims_output",
        datasets=["Cifar10"],
        substitute_dataset=["ImageNet"],
        victim_output_types=["softmax", "one_hot", "round-1"],
        substitute_model_archs=["simplenet"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("copycat", "", "All"),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("blackbox-ripper", "random", 20000),
            AttackInfo("blackbox-ripper", "optimized", 20000),
            AttackInfo("blackbox-ripper", "random", 120000),
            AttackInfo("blackbox-ripper", "optimized", 120000),
        ],
    ),
    TestSettings(
        "influence_of_the_adversary_dataset_with_less_diversity_on_the_attacks",
        datasets=["FashionMNIST"],
        substitute_dataset=["Cifar100"],
        victim_output_types=["softmax"],
        substitute_model_archs=["simplenet"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("copycat", "", "All"),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
            AttackInfo("blackbox-ripper", "random", 20000),
            AttackInfo("blackbox-ripper", "optimized", 20000),
        ],
    ),
)

ATTACKS_DICT = {
    "active-thief": ActiveThief,
    "blackbox": BlackBox,
    "copycat": CopyCat,
    "knockoff-nets": KnockOff,
    "blackbox-ripper": Ripper,
}
ATTACKS_CONFIG = {
    "active-thief": {
        "iterations": 10,
        "save_samples": True,
        "val_size": 0,
        "centers_per_iteration": 5,
        "bounds": BOUNDS,
    },
    "blackbox": {"iterations": 6, "bounds": BOUNDS},
    "copycat": {},
    "knockoff-nets": {"save_samples": True},
    "blackbox-ripper": {},
}


def get_args():
    parser = ArgumentParser(description="Model extraction attacks STL10 test")
    parser.add_argument(
        "--generator_checkpoint_cifar100",
        type=str,
        help="Path to checkpoint for Cifar100 SNGAN from Mimicry",
    )
    parser.add_argument(
        "--generator_checkpoint_imagenet",
        type=str,
        help="Path to checkpoint for ImageNet SNGAN from Mimicry",
    )
    parser.add_argument(
        "--fashion_mnist_dir",
        default="./cache/data",
        type=str,
        help="Path to Fashion-MNIST dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--cifar10_dir",
        default="./cache/data",
        type=str,
        help="Path to Cifar10 dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--cifar100_dir",
        default="./cache/data",
        type=str,
        help="Path to Cifar100 dataset (Default: ./cache/data)",
    )
    parser.add_argument(
        "--gtsrb_dir",
        default="./cache/data",
        type=str,
        help="Path to GTSRB dataset (Default: ./cache/data)",
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
        help="Path where the attacks files should be saved (Default: ./cache/)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Whether to use gpu, should be set to the number of the target device. (Default: None)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used in loaders (Default: 1)",
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


def _get_train_test_split(dataset_info: DatasetInfo) -> Tuple[Dataset, Dataset]:
    train_kwargs = {"transform": dataset_info.train_transform}
    test_kwargs = {"transform": dataset_info.test_transform, "train": False}
    if dataset_info.name == "OIModular":
        train_kwargs["seed"] = SEEDS[0]
        test_kwargs["seed"] = SEEDS[0]
        train_kwargs["download"] = (True,)
        train_kwargs["num_classes"] = 5
        dataset_dir = args.oimodular_dir
    elif dataset_info.name == "Indoor67":
        dataset_dir = args.indoor67_dir
    elif dataset_info.name == "Cifar10":
        dataset_dir = args.cifar10_dir
    elif dataset_info.name == "FashionMnist":
        dataset_dir = args.fashion_mnist_dir
    elif dataset_info.name == "GTSRB":
        dataset_dir = args.gtsrb_dir
    else:
        train_kwargs["seed"] = SEEDS[0]
        test_kwargs["seed"] = SEEDS[0]
        dataset_dir = args.caltech256_dir

    train_set = dataset_info.class_(dataset_dir, **train_kwargs)
    test_set = dataset_info.class_(dataset_dir, **test_kwargs)

    return train_set, test_set


def _prepare_victim_model(
    dataset_info: DatasetInfo, train_set: Dataset, test_set: Dataset
) -> torch.nn.Module:
    train_set, val_set = split_dataset(train_set, 0.2)

    victim_model = SimpleNet(
        num_classes=dataset_info.num_classes, dims=dataset_info.sample_dims
    )
    train_victim_model(
        victim_model,
        torch.optim.SGD,
        torch.nn.functional.cross_entropy,
        train_set,
        dataset_info.num_classes,
        VICT_TRAINING_EPOCHS,
        BATCH_SIZE,
        args.num_workers,
        optimizer_args={"lr": 0.1, "momentum": 0.9, "nesterov": True},
        val_set=val_set,
        test_set=test_set,
        evaluation_frequency=1,
        patience=20,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_args={"step_size": 60},
        gpu=args.gpu,
        save_loc=Path(args.save_loc).joinpath(
            "Attack-tests",
            f"dataset:{dataset_info.name}",
            "victim_model",
            f"sample_dims:{dataset_info.sample_dims}",
        ),
        debug=args.debug,
    )

    return victim_model


def _prepare_models_for_attack(
    dataset_info: DatasetInfo, victim_output_type: str
) -> Dict[str, Any]:
    # Prepare models for the attack
    decimals = None
    if "round" in victim_output_type:
        victim_output_type, decimals = victim_output_type.split("-")

    kwargs: Dict[str, Any] = {
        "victim_model": VictimModel(
            victim_model,
            dataset_info.num_classes,
            victim_output_type,
            int(decimals) if decimals is not None else 0,
        )
    }

    # Prepare substitute model
    substitute_model: torch.nn.Module
    if "resnet" in substitute_model_arch:
        substitute_model = ResNet(
            substitute_model_arch,
            num_classes=dataset_info.num_classes,
            smaller_resolution=True if dataset_info.sample_dims[-1] == 128 else False,
        )
    elif substitute_model_arch == "genericCNN":
        substitute_model = GenericCNN(
            dims=dataset_info.sample_dims,
            num_classes=dataset_info.num_classes,
            dropout_keep_prob=0.2,
        )
    else:
        substitute_model = SimpleNet(
            dims=dataset_info.sample_dims, num_classes=dataset_info.num_classes
        )

    substitute_optimizer = torch.optim.Adam
    substitute_loss = soft_cross_entropy
    kwargs["substitute_model"] = TrainableModel(
        substitute_model,
        dataset_info.num_classes,
        substitute_optimizer,
        substitute_loss,
    )
    return kwargs


def _add_attack_specific_kwargs(
    attack_info: AttackInfo,
    dataset_info: DatasetInfo,
    substitute_dataset: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    # Add attack specific key-name arguments
    if attack_info.name == "active-thief":
        kwargs["selection_strategy"] = attack_info.type
        kwargs["budget"] = attack_info.budget
    elif attack_info.name == "knockoff-nets":
        sampling_strategy, reward_type = attack_info.type.split("-")
        kwargs["sampling_strategy"] = sampling_strategy
        kwargs["reward_type"] = reward_type
        kwargs["budget"] = attack_info.budget
        kwargs["online_optimizer"] = torch.optim.SGD(
            kwargs["substitute_model"].parameters(), lr=0.0005, momentum=0.5
        )
    elif attack_info.name == "blackbox-ripper":
        kwargs["generated_data"] = attack_info.type
        kwargs["batches_per_epoch"] = math.floor(
            (attack_info.budget / SUB_TRAINING_EPOCHS) / BATCH_SIZE
        )
        transform = None
        if dataset_info.name == "FashionMnist":
            transform = T.Compose([T.Grayscale()])
        if substitute_dataset == "Cifar100":
            generator = Sngan(
                args.generator_checkpoint_cifar100,
                resolution=32,
                transform=transform,
            )
        else:
            generator = Sngan(
                args.generator_checkpoint_imagenet,
                resolution=32,
                transform=transform,
            )
        kwargs["generator"] = Generator(generator, latent_dim=128)

    return kwargs


def _prepare_adversary_dataset(
    attack_info: AttackInfo,
    attack_instance: AttackBase,
    dataset_info: DatasetInfo,
    substitute_dataset: str,
) -> Dict[str, Any]:
    substitute_datasets = substitute_dataset.split("-")
    substitute_data = []
    for substitute_dataset_ in substitute_datasets:
        if "ImageNet" in substitute_dataset_:
            imagenet_train_120k = ImageNet1000(
                root=args.imagenet_dir,
                transform=dataset_info.substitute_dataset_transform,
                size=120000,
                seed=SEEDS[0],
            )
            substitute_data.append(imagenet_train_120k)
        elif "STL10" in substitute_dataset_:
            stl10_train = Stl10(
                root=args.stl10_dir,
                download=True,
                transform=dataset_info.substitute_dataset_transform,
            )
            stl10_test = Stl10(
                root=args.stl10_dir,
                download=True,
                train=False,
                transform=dataset_info.substitute_dataset_transform,
            )
            substitute_data.append(stl10_train)
            substitute_data.append(stl10_test)
        elif "Cifar100" in substitute_dataset_:
            cifar100_train = Cifar100(
                root=args.cifar100_dir,
                download=True,
                transform=dataset_info.substitute_dataset_transform,
            )
            cifar100_test = Cifar100(
                root=args.cifar100_dir,
                train=False,
                download=True,
                transform=dataset_info.substitute_dataset_transform,
            )
            substitute_data.append(cifar100_train)
            substitute_data.append(cifar100_test)
        else:
            raise ValueError(
                "Only {Imagenet, STL10, Cifar100} supported as adversary dataset!"
            )

    if substitute_dataset != "ImageNet":
        substitute_data = ConcatDataset(substitute_data)
        if substitute_dataset == "ImageNet-STL10":
            substitute_data.num_classes = 1010
            substitute_data.datasets[1].targets = [
                y + 1000 for y in substitute_data.datasets[1].targets
            ]
        elif substitute_dataset == "Cifar100":
            substitute_data.num_classes = 100
        substitute_data.targets = []
        substitute_data.targets.extend(substitute_data.datasets[0].targets)
        substitute_data.targets.extend(substitute_data.datasets[1].targets)
    else:
        substitute_data = substitute_data[0]

    if attack_info.name != "blackbox-ripper":
        datasets = {"sub_data": substitute_data}

        if attack_info.name == "copycat":
            if attack_info.budget != "All":
                idxs_all = np.arange(len(datasets["sub_data"]))
                idxs_sub = np.random.permutation(idxs_all)[: attack_info.budget]
                datasets["sub_data"] = Subset(datasets["sub_data"], idxs_sub)
        elif attack_info.name == "blackbox":
            # We need to calculate initial seed size
            seed_size = math.floor(
                attack_info.budget / ((2 ** attack_instance.attack_settings.iterations))
            )
            if dataset_info.name == "Cifar10":
                samples_per_class = seed_size // dataset_info.num_classes
                if "ImageNet" in substitute_dataset:
                    # For Cifar10 we take same number of samples from these classes:
                    # 372 - plane, 274 - sports_car, 415 - hummingbird, 55 - tiger_cat, 162 - watter_buffalo,
                    # 115 - Appenzeller, 499 - bullfrog, 39 - sorrel, 237 - speedboat, 283 - trailer_truck
                    classes = [372, 273, 415, 55, 162, 115, 499, 39, 237, 283]
                idxs_sub = []
                for class_ in classes:
                    idxs_class = np.where(np.array(substitute_data.targets) == class_)[
                        0
                    ]
                    idxs_sub.append(
                        np.random.permutation(idxs_class)[:samples_per_class]
                    )
                idxs_sub = np.concatenate(idxs_sub)
            else:
                idxs_all = np.arange(len(datasets["sub_data"]))
                idxs_sub = np.random.permutation(idxs_all)[:seed_size]

            datasets["sub_data"] = Subset(datasets["sub_data"], idxs_sub)
    else:
        datasets = {}

    return datasets


def _perform_attack(
    dataset_info: DatasetInfo,
    subsitute_dataset: str,
    victim_output_type: str,
    substitute_model_arch: str,
    attack_info: AttackInfo,
    seed: int,
):
    save_loc = Path(args.save_loc)
    attack_save_loc = save_loc.joinpath(
        "Attack-tests",
        f"dataset:{dataset_info.name}",
        f"substitute_dataset:{subsitute_dataset}",
        f"victim_output_type:{victim_output_type}",
        f"sample_dims:{dataset_info.sample_dims}",
        f"sub_arch:{substitute_model_arch}",
        f"seed:{seed}",
        f"attack:{attack_info}",
    )

    # Check if the attack is already done
    final_substitute_model = attack_save_loc.joinpath(
        "substitute", "final_substitute_model-state_dict.pt"
    )
    if final_substitute_model.exists():
        print(f"{final_substitute_model} exists. Skipping the attack")
        return

    # Prepare attack arguments
    models = _prepare_models_for_attack(dataset_info, victim_output_type)
    kwargs = {**models, **ATTACKS_CONFIG[attack_info.name]}
    kwargs = _add_attack_specific_kwargs(
        attack_info, dataset_info, substitute_dataset, kwargs
    )

    attack_instance = ATTACKS_DICT[attack_info.name](**kwargs)

    # Base settings
    attack_instance.base_settings.save_loc = attack_save_loc
    attack_instance.base_settings.gpu = args.gpu
    attack_instance.base_settings.num_workers = args.num_workers
    attack_instance.base_settings.batch_size = BATCH_SIZE
    attack_instance.base_settings.seed = seed
    attack_instance.base_settings.deterministic = True
    attack_instance.base_settings.debug = args.debug

    # Trainer settings
    attack_instance.trainer_settings.training_epochs = SUB_TRAINING_EPOCHS
    attack_instance.trainer_settings.precision = args.precision
    attack_instance.trainer_settings.use_accuracy = False

    datasets = _prepare_adversary_dataset(
        attack_info, attack_instance, dataset_info, substitute_dataset
    )
    datasets["test_set"] = test_set

    attack_instance(**datasets)

    return


if __name__ == "__main__":
    args = get_args()

    for test_setting in TEST_SETTINGS:
        print(f"Test:{test_setting.name}")
        for dataset_name in test_setting.datasets:
            dataset_info = DATASET_INFOS[dataset_name]
            seed_everything(SEEDS[0])
            train_set, test_set = _get_train_test_split(dataset_info)
            victim_model = _prepare_victim_model(dataset_info, train_set, test_set)
            for victim_output_type in test_setting.victim_output_types:
                for attack_info in test_setting.attacks_to_run:
                    for substitute_dataset in test_setting.substitute_dataset:
                        for seed in SEEDS:
                            for (
                                substitute_model_arch
                            ) in test_setting.substitute_model_archs:
                                _perform_attack(
                                    dataset_info,
                                    substitute_dataset,
                                    victim_output_type,
                                    substitute_model_arch,
                                    attack_info,
                                    seed,
                                )
