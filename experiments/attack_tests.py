import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import torch
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks import ActiveThief, BlackBox, CopyCat, KnockOff, Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets.vision import ImageNet1000, Stl10
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import SimpNet

IMAGENET_TRAIN_SIZE = 100000
IMAGENET_VAL_SIZE = 20000
DIMS = (3, 64, 64)
NUM_CLASSES = 10
TRAINING_EPOCHS = 1000
PATIENCE = 100
BATCH_SIZE = 100
EVALUATION_FREQUENCY = 1

SEED = 200916

BUDGETS = [5000, 10000, 15000, 20000]
OUTPUT_TYPES = ["softmax", "one_hot"]

AttackInfo = namedtuple("AttackInfo", ["name", "type"])
ME_ATTACKS = [
    AttackInfo("active-thief", "entropy"),
    AttackInfo("active-thief", "k-center"),
    AttackInfo("active-thief", "dfal"),
    AttackInfo("active-thief", "dfal+k-center"),
    AttackInfo("blackbox", ""), AttackInfo("copycat", ""),
    AttackInfo("knockoff", "nets_adaptive-cert"),
    AttackInfo("knockoff", "nets_adaptive-div"),
    AttackInfo("knockoff", "nets_adaptive-loss"),
    AttackInfo("knockoff", "nets_adaptive-all")]
ATTACKS_DICT = {"active-thief": ActiveThief, "blackbox": BlackBox,
                "copycat": CopyCat, "knockoff-nets": KnockOff,
                "ripper": Ripper}
ATTACKS_CONFIG = {"active-thief": {"iterations": 10, "save_samples": True},
                  "blackbox": {"iterations": 6},
                  "copycat": {},
                  "knockoff-nets": {"save_samples": True},
                  "ripper": {}}


def getr_args():
    parser = ArgumentParser(description="Model extraction attacks STL10 test")
    parser.add_argument("--stl10_dir", default="./cache/data", type=str,
                        help="Location where STL10 dataset is or should be "
                             "downloaded to (Default: ./cache/data)")
    parser.add_argument("--imagenet_dir", type=str, help="Path to ImageNet "
                                                         "dataset")
    parser.add_argument("--save_loc", type=str, default="./cache/",
                        help="Path where the attacks file should be "
                             "saved (Default: ./cache/)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of gpus to be used (Default: 0)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to be used in loaders ("
                             "Default: 1)")
    parser.add_argument("--precision", default=32, type=int,
                        help="Precision of caluclation in bits must be "
                             "one of {16, 32} (Default: 32)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (Default: False)")

    return parser.parse_args()


if __name__ == "__main__":
    args = getr_args()
    seed_everything(SEED)

    save_loc = Path(args.save_loc)

    stl10_train_transform = T.Compose([T.Resize(DIMS[1:2]),
                                       T.RandomCrop(DIMS[1], padding=4),
                                       T.RandomHorizontalFlip(), T.ToTensor(),
                                       T.Normalize((0.5,), (0.5,))])
    stl10_test_transform = T.Compose([T.Resize(DIMS[1:2]), T.ToTensor(),
                                      T.Normalize((0.5,), (0.5,))])

    train_set = Stl10(args.stl10_dir, download=True,
                      transform=stl10_train_transform)
    test_set = Stl10(args.stl10_dir, split="test", download=True,
                     transform=stl10_test_transform)

    # Prepare victim model
    victim_model = SimpNet(num_classes=NUM_CLASSES)
    victim_optimizer = torch.optim.Adam(victim_model.parameters())
    victim_loss = torch.nn.functional.cross_entropy
    train_victim_model(victim_model, victim_optimizer, victim_loss, train_set,
                       NUM_CLASSES, TRAINING_EPOCHS, BATCH_SIZE,
                       args.num_workers, test_set=test_set, gpus=args.gpus,
                       save_loc=save_loc.joinpath("Attack-tests", "STL10"),
                       debug=args.debug)

    imagenet_transform = T.Compose([T.Resize(DIMS[1:3]), T.ToTensor()])
    imagenet_train = ImageNet1000(root=args.imagenet_dir,
                                  size=IMAGENET_TRAIN_SIZE,
                                  transform=imagenet_transform, seed=SEED)
    imagenet_val = ImageNet1000(root=args.imagenet_dir, train=False,
                                size=IMAGENET_VAL_SIZE,
                                transform=imagenet_transform, seed=SEED)

    substitute_loss = soft_cross_entropy
    for attack in ME_ATTACKS:
        for output_type in OUTPUT_TYPES:
            for budget in BUDGETS:
                # Prepare models for the attack
                kwargs = {"victim_model": VictimModel(victim_model,
                                                      NUM_CLASSES,
                                                      output_type)}
                substitute_model = SimpNet(num_classes=NUM_CLASSES)
                substitute_optimizer = torch.optim.Adam(
                        substitute_model.parameters())
                kwargs["substitute_model"] = TrainableModel(substitute_model,
                                                            NUM_CLASSES,
                                                            substitute_optimizer,
                                                            substitute_loss)

                kwargs.update(ATTACKS_CONFIG[attack.name])

                # Add attack specific key-name arguments
                if attack.name == "active-thief":
                    kwargs["selection_strategy"] = attack.type
                    kwargs["budget"] = budget
                elif attack.name == "knockoff":
                    sampling_strategy, reward_type = attack.type.split('-')
                    kwargs["sampling_strategy"] = sampling_strategy
                    kwargs["reward_type"] = reward_type
                    kwargs["budget"] = budget
                elif attack.name == "blackbox":
                    kwargs["budget"] = budget

                attack_instance = ATTACKS_DICT[attack.name](**kwargs)

                save_loc = f"{args.save_loc}/Attack-tests/STL10/budget" \
                           f"-{budget}/{attack.name}/{attack.type}"
                # Base settings
                attack_instance.base_settings.save_loc = Path(save_loc)
                attack_instance.base_settings.gpus = args.gpus
                attack_instance.base_settings.num_workers = args.num_workers
                attack_instance.base_settings.batch_size = BATCH_SIZE
                attack_instance.base_settings.seed = SEED
                attack_instance.base_settings.deterministic = True
                attack_instance.base_settings.debug = args.debug

                # Trainer settings
                attack_instance.trainer_settings.training_epochs = \
                    TRAINING_EPOCHS
                attack_instance.trainer_settings.patience = PATIENCE
                attack_instance.trainer_settings.evaluation_frequency = \
                    EVALUATION_FREQUENCY
                attack_instance.trainer_settings.precision = args.precision
                attack_instance.trainer_settings.use_accuracy = False

                run_kwargs = {
                    "sub_data": ConcatDataset([imagenet_train, imagenet_val]),
                    "test_set": test_set}

                attack_instance(**run_kwargs)
