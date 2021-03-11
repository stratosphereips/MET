import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import Subset
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.blackbox import BlackBox
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets.vision import Mnist
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.models.vision import GenericCNN

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 15
DIMS = (1, 28, 28)
BOUNDS = (0, 1)


def set_up(args):
    seed_everything(args.seed)

    victim_model = GenericCNN(
        dims=DIMS,
        num_classes=NUM_CLASSES,
        conv_out_channels=(32, 64),
        convs_in_block=1,
        fc_layers=(100,),
    )
    substitute_model = GenericCNN(
        dims=DIMS,
        num_classes=NUM_CLASSES,
        conv_out_channels=(32, 64),
        convs_in_block=1,
        fc_layers=(100,),
    )

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = T.Compose([T.Resize(DIMS[-1]), T.ToTensor()])
    mnist = dict()
    mnist["train"] = Mnist(root=args.mnist_dir, transform=transform, download=True)
    mnist["test"] = Mnist(
        root=args.mnist_dir, train=False, transform=transform, download=True
    )

    # Get from each class same number of samples
    targets = np.array(mnist["test"].targets)
    idx_adversary = []
    for cls in range(NUM_CLASSES):
        idx_cls = np.where(targets == cls)[0]
        idx_selected = np.random.permutation(idx_cls)[:SAMPLES_PER_CLASS]

        idx_adversary.extend(idx_selected)

    idx_test = np.random.permutation(len(mnist["test"]))
    idx_test = np.setdiff1d(idx_test, idx_adversary)

    thief_dataset = Subset(mnist["test"], idx_adversary)
    test_set = Subset(mnist["test"], idx_test)

    optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.01, momentum=0.9)
    loss = F.cross_entropy

    train_victim_model(
        victim_model,
        optimizer,
        loss,
        mnist["train"],
        NUM_CLASSES,
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_set=test_set,
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpus=args.gpus,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )

    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="labels")
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.SGD(substitute_model.parameters(), lr=0.01, momentum=0.9),
        F.cross_entropy,
    )

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    parser = BlackBox.get_attack_args()
    parser.add_argument(
        "--mnist_dir",
        default="./data/",
        type=str,
        help="Path to MNIST dataset (Default: ./data/)",
    )
    args = parser.parse_args()
    args.training_epochs = 10
    args.lmbda = 64/255

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, thief_dataset, test_set = set_up(args)
    bb = BlackBox(victim_model, substitute_model, BOUNDS, args.iterations,
                  args.lmbda)

    # Baset settings
    bb.base_settings.save_loc = Path(args.save_loc)
    bb.base_settings.gpus = args.gpus
    bb.base_settings.num_workers = args.num_workers
    bb.base_settings.batch_size = args.batch_size
    bb.base_settings.seed = args.seed
    bb.base_settings.deterministic = args.deterministic
    bb.base_settings.debug = args.debug

    # Trainer settings
    bb.trainer_settings.training_epochs = args.training_epochs
    bb.trainer_settings.precision = args.precision
    bb.trainer_settings.use_accuracy = args.accuracy

    bb(thief_dataset, test_set)
