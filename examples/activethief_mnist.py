import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from mef.utils.config import get_default_parser

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.activethief import ActiveThief
from mef.datasets.vision.imagenet1000 import ImageNet1000
from mef.models.vision.simplenet import SimpleNet
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

IMAGENET_SUBSET_SIZE = 120000
DIMS = (1, 28, 28)


def activethief_parse_args():
    description = "Activethief model extraction attack - Mnist example"
    parser = get_default_parser(description)

    parser.add_argument("-c", "--selection_strategy", default="entropy",
                        type=str, help="Activethief selection strategy can "
                                       "be one of {random, entropy, k-center, "
                                       "dfal, dfal+k-center} (Default: "
                                       "entropy)")
    parser.add_argument("-m", "--mnist_dir", default="./data/", type=str,
                        help="Path to MNIST dataset (Default: ./data/")
    parser.add_argument("-i", "--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    parser.add_argument("-o", "--iterations", default=10, type=int,
                        help="Number of iterations of the attacks (Default: "
                             "10)")
    parser.add_argument("-p", "--output_type", default="softmax", type=str,
                        help="Type of output from victim model {softmax, "
                             "logits, one_hot} (Default: softmax)")
    parser.add_argument("-z", "--init_seed_size", default=2000, type=int,
                        help="Size of the initial random query set (Default: "
                             "2000)")
    parser.add_argument("-q", "--budget", default=20000, type=int,
                        help="Size of the budget (Default: 20000)")
    args = parser.parse_args()

    return args


def set_up(args):
    victim_model = SimpleNet(input_dimensions=DIMS, num_classes=10)
    substitute_model = SimpleNet(input_dimensions=DIMS, num_classes=10)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    print("Preparing data")
    transform = transforms.Compose([transforms.CenterCrop(DIMS[2]),
                                    transforms.ToTensor()])
    mnist = dict()
    mnist["train"] = MNIST(root=args.mnist_dir, download=True,
                           transform=transform)
    mnist["test"] = MNIST(root=args.mnist_dir, train=False, download=True,
                          transform=transform)
    test_set = mnist["test"]
    train_set = mnist["train"]

    transform = transforms.Compose([transforms.CenterCrop(DIMS[2]),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    imagenet = ImageNet1000(root=args.imagenet_dir, transform=transform)
    idxs = np.random.permutation(len(imagenet))[:IMAGENET_SUBSET_SIZE]
    thief_dataset = Subset(imagenet, idxs)

    try:
        saved_model = torch.load(args.save_loc +
                                 "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters())
        loss = F.cross_entropy

        train_set, val_set = split_dataset(train_set, args.val_size)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=args.batch_size)

        mef_model = MefModule(victim_model, optimizer, loss)
        trainer = get_trainer(args.gpus, args.victim_train_epochs,
                              early_stop_tolerance=args.early_stop_tolerance,
                              save_loc=args.save_loc + "/victim/")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   args.save_loc + "/victim/final_victim_model.pt")

    return victim_model, substitute_model, thief_dataset, test_set


if __name__ == "__main__":
    args = activethief_parse_args()

    mkdir_if_missing(args.save_loc)
    victim_model, substitute_model, thief_dataset, test_set = set_up(args)

    af = ActiveThief(victim_model, substitute_model, 10, args.iterations,
                     args.selection_strategy, args.output_type,
                     args.init_seed_size, args.budget, args.training_epochs,
                     args.early_stop_tolerance, args.evaluation_frequency,
                     args.val_size, args.batch_size, args.save_loc,
                     args.gpus, args.seed, args.deterministic, args.debug)

    af.run(thief_dataset, test_set)
