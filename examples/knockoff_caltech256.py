import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from mef.attacks.knockoff import KnockOff
from mef.datasets.vision.caltech256 import Caltech256
from mef.datasets.vision.imagenet1000 import ImageNet1000
from mef.models.vision.resnet import ResNet
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset

SAMPLING_STRATEGY = "adaptive"
REWARD_TYPE = "all"
SEED = 0
DATA_DIR = None  # Define path to Caltech256 dataset
IMAGENET_DIR = None  # Define path to Imagenet2012 dataset
SAVE_LOC = "./cache/knockoff/CALTECH256"
VICT_TRAIN_EPOCHS = 200
GPUS = 1
DIMS = (3, 224, 224)


def set_up():
    victim_model = ResNet(resnet_type="resnet_34", num_classes=256)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=256)

    if GPUS:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [transforms.CenterCrop(DIMS[2]), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]

    train_set = Caltech256(DATA_DIR, transform=transform, seed=SEED)
    test_set = Caltech256(DATA_DIR, train=False, transform=transform,
                          seed=SEED)
    sub_dataset = ImageNet1000(IMAGENET_DIR, transform=transform, seed=SEED)

    # Train secret model
    try:
        saved_model = torch.load(SAVE_LOC + "/victim/final_victim_model.pt")
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
                                      batch_size=64)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=64)

        mef_model = MefModule(victim_model, optimizer, loss, lr_scheduler)
        trainer = get_trainer(GPUS, VICT_TRAIN_EPOCHS, early_stop_tolerance=10,
                              save_loc=SAVE_LOC + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                num_classes=256), sub_dataset, test_set


if __name__ == "__main__":
    mkdir_if_missing(SAVE_LOC)

    attack_variables, sub_dataset, test_set = set_up()
    ko = KnockOff(**attack_variables, sampling_strategy=SAMPLING_STRATEGY,
                  reward_type=REWARD_TYPE, save_loc=SAVE_LOC, gpus=GPUS,
                  seed=SEED)
    ko.run(sub_dataset, test_set)
