import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from mef.attacks.knockoff import KnockOff
from mef.models.vision.resnet import ResNet
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import mef
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset, split_dataset

SAMPLING_STRATEGY = "adaptive"
REWARD_TYPE = "cert"
SEED = 0
DATA_DIR = "E:/Datasets/Caltech256/256_ObjectCategories/"
SAVE_LOC = "./cache/knockoff/CALTECH256"
VICT_TRAIN_EPOCHS = 200
GPUS = 1
DIMS = (3, 224, 224)
# Reccomnded value from dataset paper
n_test = 25


def prepare_caltech256():
    try:
        x_train = np.load(DATA_DIR + "caltech256_x_train.npy", mmap_mode='r')
        y_train = np.load(DATA_DIR + "caltech256_y_train.npy", mmap_mode='r')
        x_test = np.load(DATA_DIR + "caltech256_x_test.npy", mmap_mode='r')
        y_test = np.load(DATA_DIR + "caltech256_y_test.npy", mmap_mode='r')
    except FileNotFoundError:
        # Imagenet standards
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = [transforms.CenterCrop(DIMS[2]), transforms.ToTensor(),
                     transforms.Normalize(mean, std)]
        caltech256_data = ImageFolder(root=DATA_DIR,
                                      transform=transforms.Compose(transform))

        loader = DataLoader(caltech256_data, batch_size=256, num_workers=4)

        x = []
        y = []
        for batch in tqdm(loader, desc="Loading dataset"):
            x_batch, y_batch = batch
            x.append(x_batch.numpy())
            y.append(y_batch.numpy())

        x = np.vstack(x)
        y = np.hstack(y)

        classes = np.unique(y)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for cls in tqdm(classes, desc="Train/Test split creation"):
            x_ = x[y == cls]
            x_ = np.random.permutation(x_)
            x_test.append(x_[:n_test])
            y_test.append(np.full(n_test, cls))
            x_train.append(x_[n_test:])
            y_train.append(np.full(len(x_) - n_test, cls))

        x_train = np.vstack(x_train)
        y_train = np.hstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.hstack(y_test)

        print("Saving splits")
        np.save(DATA_DIR + "caltech256_x_train.npy", x_train)
        np.save(DATA_DIR + "caltech256_y_train.npy", y_train)
        np.save(DATA_DIR + "caltech256_x_test.npy", x_test)
        np.save(DATA_DIR + "caltech256_y_test.npy", y_test)

        x_train = np.load(DATA_DIR + "caltech256_x_train.npy", mmap_mode='r')
        y_train = np.load(DATA_DIR + "caltech256_y_train.npy", mmap_mode='r')
        x_test = np.load(DATA_DIR + "caltech256_x_test.npy", mmap_mode='r')
        y_test = np.load(DATA_DIR + "caltech256_y_test.npy", mmap_mode='r')

    return x_train, y_train, x_test, y_test


def set_up():
    victim_model = ResNet(resnet_type="resnet_34", num_classes=256)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=256)

    if GPUS:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    x_train, y_train, x_test, y_test = prepare_caltech256()

    x_sub = x_train
    y_sub = y_train

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

        data = CustomDataset(x_train, y_train)
        train_set, val_set = split_dataset(data, 0.2)

        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      pin_memory=True, batch_size=64)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    batch_size=64)

        mef_model = MefModule(victim_model, optimizer, loss, lr_scheduler)
        trainer = get_trainer(GPUS, VICT_TRAIN_EPOCHS, early_stop_tolerance=10,
                              save_loc=SAVE_LOC + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   SAVE_LOC + "/victim/final_victim_model.pt")

    return dict(victim_model=victim_model, substitute_model=substitute_model,
                x_test=x_test, y_test=y_test, num_classes=256), x_sub, y_sub


if __name__ == "__main__":
    mef.Test(gpus=GPUS, seed=SEED)
    mkdir_if_missing(SAVE_LOC)

    attack_variables, x_sub, y_sub = set_up()
    KnockOff(**attack_variables, sampling_strategy=SAMPLING_STRATEGY,
             reward_type=REWARD_TYPE, save_loc=SAVE_LOC).run(x_sub, y_sub)
