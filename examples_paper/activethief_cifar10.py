import os
import sys

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.activethief import ActiveThief
from mef.attacks.base import BaseSettings, TrainerSettings
from mef.utils.pytorch.datasets.vision import ImageNet1000
from mef.utils.pytorch.models.vision.at_cnn import AtCnn
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.CenterCrop(DIMS[1]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_set = CIFAR10(root=args.cifar10_dir, download=True,
                        transform=transform)
    test_set = CIFAR10(root=args.cifar10_dir, train=False, download=True,
                       transform=transform)

    transform = transforms.Compose([transforms.CenterCrop(DIMS[1]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    imagenet_train = ImageNet1000(root=args.imagenet_dir,
                                  size=IMAGENET_TRAIN_SIZE,
                                  transform=transform, seed=args.seed)
    imagenet_val = ImageNet1000(root=args.imagenet_dir, train=False,
                                size=IMAGENET_VAL_SIZE, transform=transform,
                                seed=args.seed)
    thief_dataset = ConcatDataset([imagenet_train, imagenet_val])

    try:
        saved_model = torch.load(args.save_loc +
                                 "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters(),
                                     weight_decay=1e-3)
        loss = F.cross_entropy

        train_set, val_set = split_dataset(train_set, 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=args.batch_size)

        mef_model = MefModule(victim_model, NUM_CLASSES, optimizer, loss)
        base_settings = BaseSettings(gpus=args.gpus, save_loc=args.save_loc)
        trainer_settings = TrainerSettings(
                training_epochs=args.substitute_train_epochs,
                patience=args.patience, precision=args.precision)
        trainer = get_trainer(base_settings, trainer_settings, "victim")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   args.save_loc + "/victim/final_victim_model.pt")

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    parser = ActiveThief.get_attack_args()
    parser.add_argument("--cifar10_dir", default="./data/", type=str,
                        help="Path to CIFAR10 dataset (Default: ./data/)")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, thief_dataset, test_set = set_up(args)
    af = ActiveThief(victim_model, substitute_model, NUM_CLASSES,
                     args.iterations, args.selection_strategy,
                     args.output_type, args.budget)

    # Baset settings
    af.base_settings.save_loc = args.save_loc
    af.base_settings.gpus = args.gpus
    af.base_settings.seed = args.seed
    af.base_settings.deterministic = args.deterministic
    af.base_settings.debug = args.debug

    # Trainer settings
    af.trainer_settings.training_epochs = args.substitute_train_epochs
    af.trainer_settings.patience = args.patience
    af.trainer_settings.evaluation_frequency = args.evaluation_frequency
    af.trainer_settings.precision = args.precision
    af.trainer_settings.accuracy = args.accuracy

    # Data settings
    af.data_settings.batch_size = args.batch_size

    af(thief_dataset, test_set)
