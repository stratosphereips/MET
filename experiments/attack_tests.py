import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset, Subset

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks import ActiveThief, BlackBox, CopyCat, KnockOff, Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision import Caltech256, ImageNet1000, Indoor67

from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import ResNet

VICT_TRAINING_EPOCHS = 200
SUB_TRAINING_EPOCHS = 100
BATCH_SIZE = 64
SEED = 200916

IMAGENET_NORMALIZATION = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
Dataset = namedtuple("Dataset", ["name", "class_", "num_classes"])

test_settings = (
    TestSettings(
        "influence_of_victim_model_training_dataset",
        datasets=[
            # Dataset("OIModular", OIModular, 5),
            Dataset("Indoor67", Indoor67, 67),
            Dataset("Caltech256", Caltech256, 256),
        ],
        subsitute_dataset_sizes=[120000],
        victim_output_types=["softmax"],
        sample_dims=[(3, 128, 128)],
        substitute_model_archs=["resnet_34"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
        ],
    ),
    TestSettings(
        "scalability_in_terms_of_sample_sizes",
        datasets=[Dataset("Indoor67", Indoor67, 67)],
        subsitute_dataset_sizes=[120000],
        victim_output_types=["softmax"],
        sample_dims=[(3, 128, 128), (3, 224, 224)],
        substitute_model_archs=["resnet_34"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
        ],
    ),
    TestSettings(
        "influence_in_terms_of_victims_output",
        datasets=[Dataset("Indoor67", Indoor67, 67)],
        subsitute_dataset_sizes=[120000],
        victim_output_types=["softmax", "one_hot"],
        sample_dims=[(3, 128, 128)],
        substitute_model_archs=["resnet_34"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
        ],
    ),
    TestSettings(
        "influence_of_the_subset_dataset_diversity_on_the_attacks",
        datasets=[Dataset("Indoor67", Indoor67, 67)],
        subsitute_dataset_sizes=[120000, "all"],
        victim_output_types=["softmax"],
        sample_dims=[(3, 128, 128)],
        substitute_model_archs=["resnet_34"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
        ],
    ),
    TestSettings(
        "influence_of_the_subset_dataset_diversity_on_the_attacks",
        datasets=[Dataset("Indoor67", Indoor67, 67)],
        subsitute_dataset_sizes=[120000],
        victim_output_types=["softmax"],
        sample_dims=[(3, 128, 128), (3, 224, 224)],
        substitute_model_archs=["resnet_18", "resnet_34", "resnet_50"],
        attacks_to_run=[
            AttackInfo("blackbox", "", 20000),
            AttackInfo("copycat", "", 20000),
            AttackInfo("active-thief", "entropy", 20000),
            AttackInfo("active-thief", "k-center", 20000),
            AttackInfo("active-thief", "dfal", 20000),
            AttackInfo("active-thief", "dfal+k-center", 20000),
            AttackInfo("knockoff-nets", "adaptive-cert", 20000),
            AttackInfo("knockoff-nets", "adaptive-div", 20000),
            AttackInfo("knockoff-nets", "adaptive-loss", 20000),
            AttackInfo("knockoff-nets", "adaptive-all", 20000),
        ],
    ),
)

ATTACKS_DICT = {
    "active-thief": ActiveThief,
    "blackbox": BlackBox,
    "copycat": CopyCat,
    "knockoff-nets": KnockOff,
    "ripper": Ripper,
}
ATTACKS_CONFIG = {
    "active-thief": {"iterations": 10, "save_samples": True, "val_size": 0},
    "blackbox": {"iterations": 6},
    "copycat": {},
    "knockoff-nets": {"save_samples": True},
    "ripper": {},
}


def getr_args():
    parser = ArgumentParser(description="Model extraction attacks STL10 test")
    parser.add_argument(
        "--oimodular_dir",
        default="./cache/data",
        type=str,
        help="Path to OpenImagesModular dataset (Default: ./cache/data)",
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
        help="Path where the attacks files should be "
        "saved (Default: "
        ""
        "./cache/)",
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

    for test_setting in test_settings:
        for dataset in test_setting.datasets:
            for sample_dims in test_setting.sample_dims:
                seed_everything(SEED)
                train_transform = T.Compose(
                    [
                        T.RandomResizedCrop(sample_dims[-1]),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        IMAGENET_NORMALIZATION,
                    ]
                )
                if sample_dims[-1] == 128:
                    test_transform = [
                        T.Resize((128, 128)),
                    ]
                else:
                    test_transform = [
                        T.Resize(256),
                        T.CenterCrop(224),
                    ]

                test_transform = T.Compose(
                    [*test_transform, T.ToTensor(), IMAGENET_NORMALIZATION,]
                )

                train_kwargs = {"transform": train_transform}
                test_kwargs = {"transform": test_transform, "train": False}
                if dataset.name == "OIModular":
                    train_kwargs["seed"] = SEED
                    test_kwargs["seed"] = SEED
                    train_kwargs["download"] = (True,)
                    train_kwargs["num_classes"] = 5
                    dataset_dir = args.oimodular_dir
                elif dataset.name == "Indoor67":
                    dataset_dir = args.indoor67_dir
                else:
                    train_kwargs["seed"] = SEED
                    test_kwargs["seed"] = SEED
                    dataset_dir = args.caltech256_dir

                train_set = dataset.class_(dataset_dir, **train_kwargs)
                test_set = dataset.class_(dataset_dir, **test_kwargs)

                # Prepare victim model
                victim_model = ResNet(
                    "resnet_34",
                    num_classes=dataset.num_classes,
                    smaller_resolution=True if sample_dims[-1] == 128 else False,
                )
                victim_optimizer = torch.optim.SGD(
                    victim_model.parameters(), lr=0.1, momentum=0.5
                )
                victim_loss = torch.nn.functional.cross_entropy
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    victim_optimizer, step_size=60
                )
                train_victim_model(
                    victim_model,
                    victim_optimizer,
                    victim_loss,
                    train_set,
                    dataset.num_classes,
                    VICT_TRAINING_EPOCHS,
                    BATCH_SIZE,
                    args.num_workers,
                    test_set=test_set,
                    lr_scheduler=lr_scheduler,
                    gpus=args.gpus,
                    save_loc=save_loc.joinpath(
                        "Attack-tests",
                        f"dataset:{dataset.name}",
                        f"victim_model",
                        f"sample_dims:{sample_dims}",
                    ),
                    debug=args.debug,
                )
                for victim_output_type in test_setting.victim_output_types:
                    for attack in test_setting.attacks_to_run:
                        for (
                            substitute_dataset_size
                        ) in test_setting.subsitute_dataset_sizes:
                            for (
                                substitute_model_arch
                            ) in test_setting.substitute_model_archs:

                                attack_save_loc = save_loc.joinpath(
                                    "Attack-tests",
                                    f"dataset:{dataset.name}",
                                    f"substitute_dataset_size:"
                                    f"{substitute_dataset_size}",
                                    f"victim_output_type" f":" f"{victim_output_type}",
                                    f"sample_dims:{sample_dims}",
                                    f"sub_arch:{substitute_model_arch}",
                                    f"attack:{attack}",
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
                                        victim_model,
                                        dataset.num_classes,
                                        victim_output_type,
                                    )
                                }
                                # Prepare substitute model
                                substitute_model = ResNet(
                                    substitute_model_arch,
                                    num_classes=dataset.num_classes,
                                    smaller_resolution=True
                                    if sample_dims[-1] == 128
                                    else False,
                                )
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
                                    kwargs["budget"] = attack.budget
                                elif attack.name == "knockoff":
                                    sampling_strategy, reward_type = attack.type.split(
                                        "-"
                                    )
                                    kwargs["sampling_strategy"] = sampling_strategy
                                    kwargs["reward_type"] = reward_type
                                    kwargs["budget"] = attack.budget
                                elif attack.name == "blackbox":
                                    kwargs["budget"] = attack.budget

                                attack_instance = ATTACKS_DICT[attack.name](**kwargs)

                                # Base settings
                                attack_instance.base_settings.save_loc = attack_save_loc
                                attack_instance.base_settings.gpus = args.gpus
                                attack_instance.base_settings.num_workers = (
                                    args.num_workers
                                )
                                attack_instance.base_settings.batch_size = BATCH_SIZE
                                attack_instance.base_settings.seed = SEED
                                attack_instance.base_settings.deterministic = True
                                attack_instance.base_settings.debug = args.debug

                                # Trainer settings
                                attack_instance.trainer_settings.training_epochs = (
                                    SUB_TRAINING_EPOCHS
                                )
                                attack_instance.trainer_settings.precision = (
                                    args.precision
                                )
                                attack_instance.trainer_settings.use_accuracy = False

                                # Prepare substitute dataset
                                if sample_dims[-1] == 128:
                                    test_transform = [
                                        T.Resize((128, 128))
                                    ]
                                else:
                                    test_transform = [
                                        T.Resize(256),
                                        T.CenterCrop(224),
                                    ]

                                imagenet_transform = T.Compose(
                                    [
                                        *test_transform,
                                        T.ToTensor(),
                                        IMAGENET_NORMALIZATION,
                                    ]
                                )
                                imagenet_train = ImageNet1000(
                                    root=args.imagenet_dir,
                                    transform=imagenet_transform,
                                    seed=SEED,
                                )
                                imagenet_val = ImageNet1000(
                                    root=args.imagenet_dir,
                                    train=False,
                                    transform=imagenet_transform,
                                    seed=SEED,
                                )

                                run_kwargs = {
                                    "sub_data": ConcatDataset(
                                        [imagenet_train, imagenet_val]
                                    ),
                                    "test_set": test_set,
                                }

                                if substitute_dataset_size != "all":
                                    idxs_all = np.arange(len(run_kwargs["sub_data"]))
                                    idxs_sub = np.random.permutation(idxs_all)[
                                        : attack.budget
                                    ]
                                    run_kwargs["sub_data"] = Subset(
                                        run_kwargs["sub_data"], idxs_sub
                                    )

                                if "copycat":
                                    idxs_all = np.arange(len(run_kwargs["sub_data"]))
                                    idxs_sub = np.random.permutation(idxs_all)[
                                        : attack.budget
                                    ]
                                    run_kwargs["sub_data"] = Subset(
                                        run_kwargs["sub_data"], idxs_sub
                                    )

                                attack_instance(**run_kwargs)
