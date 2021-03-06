import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from met.attacks.copycat import CopyCat
from met.utils.experiment import train_victim_model
from met.utils.ios import mkdir_if_missing
from met.utils.pytorch.datasets.vision import Cifar10, ImageNet1000, Stl10
from met.utils.pytorch.lightning.module import TrainableModel, VictimModel
from met.utils.pytorch.models.vision import Vgg

NUM_CLASSES = 9


class GOCData:
    def __init__(self, imagenet_dir, stl10_dir, cifar10_dir):
        self.npd_size = 120000
        self.imagenet_dir = imagenet_dir
        self.stl10_dir = stl10_dir
        self.cifar10_dir = cifar10_dir
        self.dims = (3, 32, 32)

        # Imagenet values
        self.transform = T.Compose(
            [T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

        self.test_set = None
        self.od_dataset = None
        self.pd_dataset = None
        self.npd_dataset = None

        self._setup()

    def _remove_class(self, class_to_remove, labels, data, class_to_idx, classes):
        class_idx = class_to_idx[class_to_remove]

        class_idxes = np.where(labels == np.atleast_1d(class_idx))
        labels = np.delete(labels, class_idxes)
        labels[labels > class_idx] -= 1

        data = np.delete(data, class_idxes, 0)

        classes.remove(class_to_remove)

        # class_to_idx update
        class_to_idx.pop(class_to_remove)
        for name, idx in class_to_idx.items():
            if idx > class_idx:
                class_to_idx[name] -= 1

        return labels.tolist(), data, class_to_idx, classes

    def _setup(self):
        cifar10 = dict()
        cifar10["train"] = Cifar10(
            self.cifar10_dir, transform=self.transform, download=True
        )
        cifar10["test"] = Cifar10(
            self.cifar10_dir, train=False, transform=self.transform, download=True
        )

        stl10 = dict()
        stl10["train"] = Stl10(self.stl10_dir, transform=self.transform, download=True)
        stl10["test"] = Stl10(
            self.stl10_dir, train=False, transform=self.transform, download=True
        )

        # Replace car with automobile to make the class name same as in the
        # cifar10
        for setx in stl10.values():
            for i, cls in enumerate(setx.classes):
                if cls == "car":
                    setx.classes[i] = "automobile"
                    setx.classes.sort()
                    break

        # Remove frog class from CIFAR-10 and monkey from STL-10 so both
        # datasets have same class
        for name, setx in cifar10.items():
            (
                setx.targets,
                setx.data,
                setx.class_to_idx,
                setx.classes,
            ) = self._remove_class(
                "frog", setx.targets, setx.data, setx.class_to_idx, setx.classes
            )

        for name, setx in stl10.items():
            stl10_class_to_idx = {
                cls: idx for cls, idx in zip(setx.classes, range(len(setx.classes)))
            }
            (
                setx.labels,
                setx.data,
                stl10_class_to_idx,
                setx.classes,
            ) = self._remove_class(
                "monkey", setx.labels, setx.data, stl10_class_to_idx, setx.classes
            )

        self.test_set = cifar10["test"]
        self.od_dataset = cifar10["train"]
        self.pd_dataset = ConcatDataset([stl10["train"], stl10["test"]])

        imagenet = ImageNet1000(self.imagenet_dir, transform=self.transform)
        idxs = np.random.permutation(len(imagenet))[: self.npd_size]
        self.npd_dataset = Subset(imagenet, idxs)


def set_up(args):
    seed_everything(args.seed)

    victim_model = Vgg(vgg_type="vgg_16", num_classes=9)
    substitute_model = Vgg(vgg_type="vgg_16", num_classes=9)

    print("Preparing data")
    goc = GOCData(args.imagenet_dir, args.cifar10_dir, args.stl10_dir)

    victim_training_epochs = 20
    train_victim_model(
        victim_model,
        torch.optim.SGD,
        F.cross_entropy,
        goc.od_dataset,
        NUM_CLASSES,
        victim_training_epochs,
        args.batch_size,
        args.num_workers,
        optimizer_args={"lr": 0.1, "momentum": 0.5},
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpu=args.gpu,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )

    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="softmax")
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.SGD,
        F.cross_entropy,
        optimizer_args={"lr": 0.1, "momentum": 0.5},
    )

    return (
        victim_model,
        substitute_model,
        [goc.npd_dataset, goc.pd_dataset],
        goc.test_set,
    )


if __name__ == "__main__":
    parser = CopyCat.get_attack_args()
    parser.add_argument(
        "--stl10_dir", default="./data", type=str, help="Path to Stl10 dataset"
    )
    parser.add_argument(
        "--cifar10_dir", default="./data", type=str, help="Path to Cifar10 dataset"
    )
    parser.add_argument("--imagenet_dir", type=str, help="Path to ImageNet dataset")
    args = parser.parse_args()
    args.training_epochs = 5

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, adversary_dataset, test_set = set_up(args)
    copycat = CopyCat(victim_model, substitute_model)

    # Baset settings
    copycat.base_settings.save_loc = Path(args.save_loc)
    copycat.base_settings.gpu = args.gpu
    copycat.base_settings.num_workers = args.num_workers
    copycat.base_settings.batch_size = args.batch_size
    copycat.base_settings.seed = args.seed
    copycat.base_settings.deterministic = args.deterministic
    copycat.base_settings.debug = args.debug

    # Trainer settings
    copycat.trainer_settings.training_epochs = args.training_epochs
    copycat.trainer_settings.precision = args.precision
    copycat.trainer_settings.use_accuracy = args.accuracy

    print("CopyCat attack with NPD dataset")
    copycat(adversary_dataset[0], test_set)

    print("CopyCat attack with PD dataset")
    copycat(adversary_dataset[1], test_set)
