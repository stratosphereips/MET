import os
import sys
from argparse import ArgumentParser

import torch
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision.oimodular import OIModular
from mef.utils.pytorch.models.vision import SimpleNet, ResNet


def getr_args():
    parser = ArgumentParser(description="OImodular experiment")
    parser.add_argument("--simplenet", action="store_true",
                        help="Use SimpleNet instead of Resnet34")
    parser.add_argument("--resolution", default=224, type=int,
                        help="Resolution of samples")
    parser.add_argument("--num_classes", default=5, type=int,
                        help="Which number of classes to use. Can be one of "
                             "{5, 17, 51} (Default: 5)")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size which should be used (Default: 64)")
    parser.add_argument("--oi-dir", default="./cache/data", type=str,
                        help="Location where OpenImagesModular dataset is or "
                             "should be located (Default: ./cache/data)")

    return parser.parse_args()


if __name__ == "__main__":
    args = getr_args()

    transform = T.Compose([T.Grayscale(num_output_channels=3),
                           T.Resize((args.resolution,)), T.ToTensor()])
    train_set = OIModular("/data/vit/openimagesv6", args.num_classes,
                          download=True, transform=transform)
    test_set = OIModular("/data/vit/openimagesv6", args.num_classes,
                         train=False, download=True, transform=transform)

    train_set, val_set = split_dataset(train_set, 0.2)

    if args.simplenet:
        test_model = SimpleNet((3, args.resolution, args.resolution),
                               args.num_classes)
    else:
        test_model = ResNet("resnet_34", args.num_classes)

    optimizer = torch.optim.Adam(test_model.parameters())
    loss = torch.nn.functional.cross_entropy

    train_victim_model(test_model, optimizer, loss, train_set,
                       args.num_classes, 1000, args.batch_size, 16,
                       val_set, test_set, gpus=1,
                       save_loc=f"./cache/OIModular{args.num_classes}/")
