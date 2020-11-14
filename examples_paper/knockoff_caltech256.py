import os
import sys

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.knockoff import KnockOff
from mef.attacks.base import BaseSettings, TrainerSettings
from mef.utils.pytorch.datasets.vision import ImageNet1000, Caltech256
from mef.utils.pytorch.models.vision import ResNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

NUM_CLASSES = 256


def set_up(args):
    seed_everything(args.seed)

    victim_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([T.CenterCrop(224), T.ToTensor(),
                           T.Normalize(mean, std)])

    train_set = Caltech256(args.caltech256_dir, transform=transform,
                           seed=args.seed)
    test_set = Caltech256(args.caltech256_dir, train=False,
                          transform=transform, seed=args.seed)
    sub_dataset = ImageNet1000(args.imagenet_dir, transform=transform,
                               seed=args.seed)

    # Train secret model
    try:
        saved_model = torch.load(args.save_loc +
                                 "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")
        optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.1,
                                    momentum=0.5)
        loss = F.cross_entropy
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60)

        train_set, val_set = split_dataset(train_set, 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=args.batch_size)

        mef_model = MefModule(victim_model, NUM_CLASSES, optimizer, loss,
                              lr_scheduler)
        base_settings = BaseSettings(gpus=args.gpus, save_loc=args.save_loc)
        trainer_settings = TrainerSettings(
                training_epochs=args.training_epochs,
                _validation=False, precision=args.precision)
        trainer = get_trainer(base_settings, trainer_settings, "victim")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   args.save_loc + "/victim/final_victim_model.pt")

    return victim_model, substitute_model, sub_dataset, test_set


if __name__ == "__main__":
    parser = KnockOff.get_attack_args()
    parser.add_argument("--caltech256_dir", default="./data/", type=str,
                        help="Path to Caltech256 dataset (Default: ./data/")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    parser.add_argument("--victim_train_epochs", default=200, type=int,
                        help="Number of epochs for which the victim should "
                             "train for (Default: 200)")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, sub_dataset, test_set = set_up(args)
    ko = KnockOff(victim_model, substitute_model, NUM_CLASSES,
                  args.sampling_strategy, args.reward_type, args.output_type,
                  args.budget)

    # Baset settings
    ko.base_settings.save_loc = args.save_loc
    ko.base_settings.gpus = args.gpus
    ko.base_settings.seed = args.seed
    ko.base_settings.deterministic = args.deterministic
    ko.base_settings.debug = args.debug

    # Trainer settings
    ko.trainer_settings.training_epochs = args.training_epochs
    ko.trainer_settings.precision = args.precision
    ko.trainer_settings.accuracy = args.accuracy

    # Data settings
    ko.data_settings.batch_size = args.batch_size

    ko(sub_dataset, test_set)
