import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.blackbox import BlackBox
from mef.attacks.base import BaseSettings, TrainerSettings
from mef.utils.pytorch.models.vision import SimpleNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

NUM_CLASSES = 10
DIMS = (1, 28, 28)


def set_up(args):
    seed_everything(args.seed)

    victim_model = SimpleNet(dims=DIMS, num_classes=NUM_CLASSES)
    substitute_model = SimpleNet(dims=DIMS, num_classes=NUM_CLASSES)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = T.Compose([T.CenterCrop(DIMS[1:]), T.ToTensor()])
    mnist = dict()
    mnist["train"] = MNIST(root=args.mnist_dir, download=True,
                           transform=transform)
    mnist["test"] = MNIST(root=args.mnist_dir, train=False, download=True,
                          transform=transform)

    idx_test = np.random.permutation(len(mnist["test"]))
    idx_sub = idx_test[:args.holdout_size]
    idx_test = idx_test[args.holdout_size:]

    thief_dataset = Subset(mnist["test"], idx_sub)
    test_set = Subset(mnist["test"], idx_test)

    # Train secret model
    try:
        saved_model = torch.load(args.save_loc +
                                 "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        train_set, val_set = split_dataset(mnist["train"], 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=args.batch_size)

        mef_model = MefModule(victim_model, NUM_CLASSES, optimizer, loss)
        base_settings = BaseSettings(gpus=args.gpus, save_loc=args.save_loc)
        trainer_settings = TrainerSettings(
                training_epochs=args.training_epochs,
                _validation=False, precision=args.precision)
        trainer = get_trainer(base_settings, trainer_settings, "victim")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   args.save_loc + "/victim/final_victim_model.pt")

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    parser = BlackBox.get_attack_args()
    parser.add_argument("--mnist_dir", default="./data/", type=str,
                        help="Path to MNIST dataset (Default: ./data/")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    parser.add_argument("--holdout_size", default=150, type=int,
                        help="Hold out size from MNIST test (Default: 150)")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, thief_dataset, test_set = set_up(args)
    bb = BlackBox(victim_model, substitute_model, NUM_CLASSES, args.iterations,
                  args.lmbda)

    # Baset settings
    bb.base_settings.save_loc = args.save_loc
    bb.base_settings.gpus = args.gpus
    bb.base_settings.seed = args.seed
    bb.base_settings.deterministic = args.deterministic
    bb.base_settings.debug = args.debug

    # Trainer settings
    bb.trainer_settings.training_epochs = args.substitute_train_epochs
    bb.trainer_settings.precision = args.precision
    bb.trainer_settings.accuracy = args.accuracy

    # Data settings
    bb.data_settings.batch_size = args.batch_size

    bb(thief_dataset, test_set)
