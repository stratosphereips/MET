import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.knockoff import KnockOff
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets.vision import ImageNet1000, Caltech256
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import ResNet

NUM_CLASSES = 256
DIMS = (3, 224, 224)


def set_up(args):
    seed_everything(args.seed)

    victim_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    transform = T.Compose([T.Resize(DIMS[1:3]), T.ToTensor()])

    train_set = Caltech256(args.caltech256_dir, transform=transform, seed=args.seed)
    test_set = Caltech256(
        args.caltech256_dir, train=False, transform=transform, seed=args.seed
    )
    sub_dataset = ImageNet1000(args.imagenet_dir, transform=transform, seed=args.seed)

    vict_optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.1, momentum=0.5)
    loss = F.cross_entropy
    lr_scheduler = torch.optim.lr_scheduler.StepLR(vict_optimizer, step_size=60)

    victim_train_epochs = 200
    train_victim_model(
        victim_model,
        vict_optimizer,
        loss,
        train_set,
        NUM_CLASSES,
        victim_train_epochs,
        args.batch_size,
        args.num_workers,
        lr_scheduler=lr_scheduler,
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpus=args.gpus,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )
    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="softmax")

    sub_optimizer = torch.optim.SGD(
        substitute_model.parameters(), lr=0.01, momentum=0.5
    )
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        sub_optimizer,
        soft_cross_entropy,
        torch.optim.lr_scheduler.StepLR(sub_optimizer, step_size=60),
    )

    # Because we are using adaptive_flat we are using the same experiment
    # setup as in the paper, where it is assumed that the attacker has access
    # to all available data
    sub_dataset = ConcatDataset([sub_dataset, train_set, test_set])
    sub_dataset.num_classes = 1256
    sub_dataset.datasets[1].targets = [
        y + 1000 for y in sub_dataset.datasets[1].targets
    ]
    sub_dataset.datasets[2].targets = [
        y + 1000 for y in sub_dataset.datasets[2].targets
    ]
    sub_dataset.targets = []
    sub_dataset.targets.extend(sub_dataset.datasets[0].targets)
    sub_dataset.targets.extend(sub_dataset.datasets[1].targets)
    sub_dataset.targets.extend(sub_dataset.datasets[2].targets)

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
        args.sampling_strategy,
        args.reward_type,
        torch.optim.SGD(self._substitute_model.parameters(), lr=0.0005, momentum=0.5),
        args.budget,
    )

    # Baset settings
    ko.base_settings.save_loc = Path(args.save_loc)
    ko.base_settings.gpus = args.gpus
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
