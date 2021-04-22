import math
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks import ActiveThief, BlackBox, CopyCat, KnockOff, Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision import (
    Caltech256,
    ImageNet1000,
    Indoor67,
    Cifar10,
    FashionMnist,
    GTSRB,
)
from mef.utils.pytorch.lighting.module import Generator, TrainableModel, VictimModel
from mef.utils.pytorch.models.generators import Sngan
from mef.utils.pytorch.models.vision import ResNet, SimpleNet, GenericCNN


VICT_TRAINING_EPOCHS = 200
SUB_TRAINING_EPOCHS = 100
BATCH_SIZE = 150
SEED = 200916
# BOUNDS = (-2.1179, 2.64)  # for foolbox attacks
# IMAGENET_NORMALIZATION = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
BOUNDS = (-1, 1)

TestSettings = namedtuple(
    "TestSettings",
    [
        "name",
        "datasets",
        "subsitute_dataset_sizes",
        "victim_output_types",
        "sample_dims",
        "substitute_model_archs",
        "attacks_to_run",
    ],
)
AttackInfo = namedtuple("AttackInfo", ["name", "type", "budget"])
DatasetInfo = namedtuple(
    "Dataset",
    [
        "name",
        "class_",
        "num_classes",
        "train_transform",
        "test_transform",
        "imagenet_transform",
    ],
)

datasets_info = {
    "FashionMnist": DatasetInfo(
        "FashionMnist",
        FashionMnist,
        10,
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
        T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    ),
    "GTSRB": DatasetInfo(
        "GTSRB",
        GTSRB,
        43,
        T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        ),
        T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    ),
}

test_settings = (
    TestSettings(
        "comparison_of_attacks_on_most_popular_datasets_from_paper",
        datasets=["FashionMnist", "Cifar10", "GTSRB"],
        subsitute_dataset_sizes=[120000],
        victim_output_types=["softmax"],
        sample_dims=[(3, 32, 32)],
        substitute_model_archs=["simplenet"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("copycat", "", 120000),
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
    # TestSettings(
    #     "influence_of_victim_model_training_dataset",
    #     datasets=[
    #         DatasetInfo("Indoor67", Indoor67, 67),
    #         DatasetInfo("Caltech256", Caltech256, 256),
    #     ],
    #     subsitute_dataset_sizes=[120000],
    #     victim_output_types=["softmax"],
    #     sample_dims=[(3, 128, 128)],
    #     substitute_model_archs=["resnet_34"],
    #     attacks_to_run=[
    #         AttackInfo("blackbox", "", 20000),
    #         AttackInfo("copycat", "", 20000),
    #         AttackInfo("active-thief", "entropy", 20000),
    #         AttackInfo("active-thief", "k-center", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-cert", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-div", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-loss", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-all", 20000),
    #     ],
    # ),
    # TestSettings(
    #     "scalability_in_terms_of_sample_sizes",
    #     datasets=[DatasetInfo("Indoor67", Indoor67, 67)],
    #     subsitute_dataset_sizes=[120000],
    #     victim_output_types=["softmax"],
    #     sample_dims=[(3, 128, 128), (3, 224, 224)],
    #     substitute_model_archs=["resnet_34"],
    #     attacks_to_run=[
    #         AttackInfo("copycat", "", 20000),
    #         AttackInfo("active-thief", "entropy", 20000),
    #         AttackInfo("active-thief", "k-center", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-cert", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-div", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-loss", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-all", 20000),
    #     ],
    # ),
    # TestSettings(
    #     "influence_in_terms_of_victims_output",
    #     datasets=[DatasetInfo("Indoor67", Indoor67, 67)],
    #     subsitute_dataset_sizes=[120000],
    #     victim_output_types=["softmax", "one_hot"],
    #     sample_dims=[(3, 128, 128)],
    #     substitute_model_archs=["resnet_34"],
    #     attacks_to_run=[
    #         AttackInfo("blackbox", "", 20000),
    #         AttackInfo("copycat", "", 20000),
    #         AttackInfo("active-thief", "entropy", 20000),
    #         AttackInfo("active-thief", "k-center", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-cert", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-div", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-loss", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-all", 20000),
    #     ],
    # ),
    # TestSettings(
    #     "influence_of_the_subset_dataset_diversity_on_the_attacks",
    #     datasets=[DatasetInfo("Indoor67", Indoor67, 67)],
    #     subsitute_dataset_sizes=[120000, "all"],
    #     victim_output_types=["softmax"],
    #     sample_dims=[(3, 128, 128)],
    #     substitute_model_archs=["resnet_34"],
    #     attacks_to_run=[
    #         AttackInfo("blackbox", "", 20000),
    #         AttackInfo("copycat", "", 20000),
    #         AttackInfo("active-thief", "entropy", 20000),
    #         AttackInfo("active-thief", "k-center", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-cert", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-div", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-loss", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-all", 20000),
    #     ],
    # ),
    # TestSettings(
    #     "influence_of_the_subset_model_on_the_attacks",
    #     datasets=[DatasetInfo("Indoor67", Indoor67, 67)],
    #     subsitute_dataset_sizes=[120000],
    #     victim_output_types=["softmax"],
    #     sample_dims=[(3, 128, 128), (3, 224, 224)],
    #     substitute_model_archs=["resnet_18", "resnet_34", "resnet_50"],
    #     attacks_to_run=[
    #         AttackInfo("copycat", "", 20000),
    #         AttackInfo("active-thief", "entropy", 20000),
    #         AttackInfo("active-thief", "k-center", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-cert", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-div", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-loss", 20000),
    #         AttackInfo("knockoff-nets", "adaptive-all", 20000),
    #     ],
    # ),
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


def getr_args():
    parser = ArgumentParser(description="Model extraction attacks STL10 test")
    parser.add_argument(
        "--generator_checkpoint_cifar10",
        type=str,
        help="Path to checkpoint for Cifar10 SNGAN from Mimicry",
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
        "--gtsrb_dir",
        default="./cache/data",
        type=str,
        help="Path to GTSRB dataset (Default: ./cache/data)",
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


def _prepare_victim_model(
    dataset: DatasetInfo, sample_dims: Tuple[int, int, int]
) -> Tuple[torch.nn.Module, Dataset]:
    # train_transform = T.Compose(
    #     [
    #         T.RandomResizedCrop(sample_dims[-1]),
    #         T.RandomHorizontalFlip(),
    #         T.ToTensor(),
    #         IMAGENET_NORMALIZATION,
    #     ]
    # )
    # if sample_dims[-1] == 128:
    #     test_transform = [
    #         T.Resize((128, 128)),
    #     ]
    # else:
    #     test_transform = [
    #         T.Resize(256),
    #         T.CenterCrop(224),
    #     ]

    # test_transform = T.Compose([*test_transform, T.ToTensor(), IMAGENET_NORMALIZATION,])

    train_kwargs = {"transform": dataset.train_transform}
    test_kwargs = {"transform": dataset.test_transform, "train": False}
    if dataset.name == "OIModular":
        train_kwargs["seed"] = SEED
        test_kwargs["seed"] = SEED
        train_kwargs["download"] = (True,)
        train_kwargs["num_classes"] = 5
        dataset_dir = args.oimodular_dir
    elif dataset.name == "Indoor67":
        dataset_dir = args.indoor67_dir
    elif dataset.name == "Cifar10":
        dataset_dir = args.cifar10_dir
    elif dataset.name == "FashionMnist":
        dataset_dir = args.fashion_mnist_dir
        sample_dims = (1, *sample_dims[1:3])
    elif dataset.name == "GTSRB":
        dataset_dir = args.gtsrb_dir
    else:
        train_kwargs["seed"] = SEED
        test_kwargs["seed"] = SEED
        dataset_dir = args.caltech256_dir

    train_set = dataset.class_(dataset_dir, **train_kwargs)
    train_set, val_set = split_dataset(train_set, 0.2)
    test_set = dataset.class_(dataset_dir, **test_kwargs)

    # TODO: make this changeable
    # Prepare victim model
    # victim_model = ResNet(
    #     "resnet_34",
    #     num_classes=dataset.num_classes,
    #     smaller_resolution=True if sample_dims[-1] == 128 else False,
    # )
    victim_model = SimpleNet(num_classes=dataset.num_classes, dims=sample_dims)
    train_victim_model(
        victim_model,
        torch.optim.SGD,
        torch.nn.functional.cross_entropy,
        train_set,
        dataset.num_classes,
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
        save_loc=save_loc.joinpath(
            "Attack-tests",
            f"dataset:{dataset.name}",
            f"victim_model",
            f"sample_dims:{sample_dims}",
        ),
        debug=args.debug,
    )

    return victim_model, test_set


def _perform_attack(
    dataset: DatasetInfo,
    substitute_dataset_size: int,
    victim_output_type: str,
    sample_dims: Tuple[int, int, int],
    substitute_model_arch: str,
    attack: str,
):
    attack_save_loc = save_loc.joinpath(
        "Attack-tests",
        f"dataset:{dataset.name}",
        f"substitute_dataset_size:{substitute_dataset_size}",
        f"victim_output_type:{victim_output_type}",
        f"sample_dims:{sample_dims}",
        f"sub_arch:{substitute_model_arch}",
        f"attack:{attack}",
    )

    # Check if the attack is already done
    final_substitute_model = attack_save_loc.joinpath(
        "substitute", "final_substitute_model-state_dict.pt"
    )
    if final_substitute_model.exists():
        print(f"{final_substitute_model} exists. Skipping the attack")
        return

    # Prepare models for the attack
    kwargs = {
        "victim_model": VictimModel(
            victim_model,
            dataset.num_classes,
            victim_output_type,
        )
    }

    if dataset.name == "FashionMnist":
        sample_dims = (1, *sample_dims[1:3])

    # Prepare substitute model
    if "resnet" in substitute_model_arch:
        substitute_model = ResNet(
            substitute_model_arch,
            num_classes=dataset.num_classes,
            smaller_resolution=True if sample_dims[-1] == 128 else False,
        )
    elif substitute_model_arch == "genericCNN":
        substitute_model = GenericCNN(
            dims=sample_dims, num_classes=dataset.num_classes, dropout_keep_prob=0.2
        )
    else:
        substitute_model = SimpleNet(dims=sample_dims, num_classes=dataset.num_classes)

    substitute_optimizer = torch.optim.Adam
    substitute_loss = soft_cross_entropy
    kwargs["substitute_model"] = TrainableModel(
        substitute_model,
        dataset.num_classes,
        substitute_optimizer,
        substitute_loss,
    )

    kwargs.update(ATTACKS_CONFIG[attack.name])

    # Add attack specific key-name arguments
    if attack.name == "active-thief":
        kwargs["selection_strategy"] = attack.type
        kwargs["budget"] = attack.budget
    elif attack.name == "knockoff-nets":
        sampling_strategy, reward_type = attack.type.split("-")
        kwargs["sampling_strategy"] = sampling_strategy
        kwargs["reward_type"] = reward_type
        kwargs["budget"] = attack.budget
        kwargs["online_optimizer"] = torch.optim.SGD(
            kwargs["substitute_model"].parameters(), lr=0.0005, momentum=0.5
        )
    elif attack.name == "blackbox-ripper":
        kwargs["generated_data"] = attack.type
        kwargs["batches_per_epoch"] = math.floor(
            (attack.budget / SUB_TRAINING_EPOCHS) / BATCH_SIZE
        )
        transform = None
        if dataset.name == "FashionMnist":
            transform = T.Compose([T.Grayscale()])

        generator = Sngan(
            args.generator_checkpoint_imagenet,
            resolution=32,
            transform=transform,
        )
        kwargs["generator"] = Generator(generator, latent_dim=128)

    attack_instance = ATTACKS_DICT[attack.name](**kwargs)

    # Base settings
    attack_instance.base_settings.save_loc = attack_save_loc
    attack_instance.base_settings.gpu = args.gpu
    attack_instance.base_settings.num_workers = args.num_workers
    attack_instance.base_settings.batch_size = BATCH_SIZE
    attack_instance.base_settings.seed = SEED
    attack_instance.base_settings.deterministic = True
    attack_instance.base_settings.debug = args.debug

    # Trainer settings
    attack_instance.trainer_settings.training_epochs = SUB_TRAINING_EPOCHS
    attack_instance.trainer_settings.precision = args.precision
    attack_instance.trainer_settings.use_accuracy = False

    # # Prepare substitute dataset
    # if sample_dims[-1] == 128:
    #     test_transform = [T.Resize((128, 128))]
    # else:
    #     test_transform = [
    #         T.Resize(256),
    #         T.CenterCrop(224),
    #     ]

    # imagenet_transform = T.Compose(
    #     [*test_transform, T.ToTensor(), IMAGENET_NORMALIZATION,]
    # )
    substitute_data = ImageNet1000(
        root=args.imagenet_dir,
        transform=dataset.imagenet_transform,
        size=substitute_dataset_size,
        seed=SEED,
    )

    if substitute_dataset_size == "all":
        pass

    if attack.name != "blackbox-ripper":
        run_kwargs = {
            "sub_data": substitute_data,
            "test_set": test_set,
        }

        if attack.budget != len(run_kwargs["sub_data"]):
            if attack.name == "copycat":
                idxs_all = np.arange(len(run_kwargs["sub_data"]))
                idxs_sub = np.random.permutation(idxs_all)[: attack.budget]
                run_kwargs["sub_data"] = Subset(run_kwargs["sub_data"], idxs_sub)
            elif attack.name == "blackbox":
                # We need to calculate initial seed size
                seed_size = math.floor(
                    attack.budget / ((2 ** attack_instance.attack_settings.iterations))
                )
                idxs_all = np.arange(len(run_kwargs["sub_data"]))
                if substitute_dataset_size == "all":
                    pass
                else:
                    idxs_sub = np.random.permutation(idxs_all)[:seed_size]
                    run_kwargs["sub_data"] = Subset(run_kwargs["sub_data"], idxs_sub)
    else:
        run_kwargs = {"test_set": test_set}

    attack_instance(**run_kwargs)

    return


if __name__ == "__main__":
    args = getr_args()

    save_loc = Path(args.save_loc)

    for test_setting in test_settings:
        print(f"Test:{test_setting.name}")
        for dataset_name in test_setting.datasets:
            dataset = datasets_info[dataset_name]
            for sample_dims in test_setting.sample_dims:
                seed_everything(SEED)
                victim_model, test_set = _prepare_victim_model(dataset, sample_dims)
                for victim_output_type in test_setting.victim_output_types:
                    for attack in test_setting.attacks_to_run:
                        for (
                            substitute_dataset_size
                        ) in test_setting.subsitute_dataset_sizes:
                            for (
                                substitute_model_arch
                            ) in test_setting.substitute_model_archs:
                                _perform_attack(
                                    dataset,
                                    substitute_dataset_size,
                                    victim_output_type,
                                    sample_dims,
                                    substitute_model_arch,
                                    attack,
                                )
