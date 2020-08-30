from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageNet
from torchvision.transforms import transforms

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.details import ModelDetails
from mef.utils.ios import mkdir_if_missing

config_file = "../config.yaml"
data_root = "../data"
imagenet_root = "E:/Datasets/ImageNet2012"
save_loc = "./cache/copycat/GOC"
target_save_loc = save_loc + "/final_target_model.pt"
opd_save_loc = save_loc + "/final_opd_model.pt"
npd_size = 2000
train_epochs = 1
device = "cuda"  # cuda or cpu


def remove_class(class_to_remove, labels, data, class_to_idx, classes):
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


def train_model(model, train_data, save_loc):
    train_loader = DataLoader(dataset=train_data, batch_size=128,
                              shuffle=True, num_workers=1, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    loss_function = F.cross_entropy
    trainer = create_supervised_trainer(model, optimizer, loss_function,
                                        device=device)

    ProgressBar().attach(trainer)
    trainer.run(train_loader, max_epochs=train_epochs)

    torch.save(dict(state_dict=model.state_dict()), save_loc)

    return


class TestCopyCat(TestCase):
    def setUp(self) -> None:
        mkdir_if_missing(save_loc)
        mef.Test(config_file)

        model_details = ModelDetails(net=dict(name="vgg_16",
                                              act="relu",
                                              drop="none",
                                              pool="max_2",
                                              ks=3,
                                              n_conv=13,
                                              n_fc=3))
        self.target_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                                model_details=model_details)
        self.opd_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                             model_details=model_details)
        self.copycat_model = Vgg(input_dimensions=(3, 64, 64), num_classes=9,
                                 model_details=model_details)

        if device == "cuda":
            self.target_model.cuda()
            self.opd_model.cuda()
            self.copycat_model.cuda()

        print("Preparing data")
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean,
                                                                                              std)]

        cifar10 = dict()
        cifar10["train"] = CIFAR10(root=data_root, download=True,
                                   transform=transforms.Compose(transform))

        cifar10["test"] = CIFAR10(root=data_root, train=False, download=True,
                                  transform=transforms.Compose(transform))

        transform = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean,
                                                                                              std)]
        stl10 = dict()
        stl10["train"] = STL10(root=data_root, transform=transforms.Compose(transform),
                               download=True)
        stl10["test"] = STL10(root=data_root, split="test",
                              transform=transforms.Compose(transform),
                              download=True)

        # Replace car with automobile to make the class name same as in the cifar10
        for setx in stl10.values():
            for i, cls in enumerate(setx.classes):
                if cls == "car":
                    setx.classes[i] = "automobile"
                    setx.classes.sort()
                    break

        transform = [transforms.Resize((64, 64)), transforms.ToTensor()]
        imagenet = dict()
        imagenet["train"] = ImageNet(root=imagenet_root, transform=transforms.Compose(transform))
        imagenet["test"] = ImageNet(root=imagenet_root, split="val",
                                    transform=transforms.Compose(transform))
        imagenet["all"] = ConcatDataset(imagenet.values())
        idx = np.arange(len(imagenet["all"]))
        npd_idx = np.random.choice(idx, size=npd_size, replace=False)

        # Remove frog class from CIFAR-10 and monkey from STL-10 so both datasets have same class
        for name, setx in cifar10.items():
            setx.targets, setx.data, setx.class_to_idx, setx.classes = remove_class("frog",
                                                                                    setx.targets,
                                                                                    setx.data,
                                                                                    setx.class_to_idx,
                                                                                    setx.classes)
        for name, setx in stl10.items():
            stl10_class_to_idx = {cls: idx for cls, idx in zip(setx.classes, range(len(
                setx.classes)))}
            setx.labels, setx.data, stl10_class_to_idx, setx.classes = remove_class("monkey",
                                                                                    setx.labels,
                                                                                    setx.data,
                                                                                    stl10_class_to_idx,
                                                                                    setx.classes)

        self.test_dataset = cifar10["test"]
        self.original_domain_dataset = cifar10["train"]
        self.problem_domain_dataset = ConcatDataset([stl10["train"], stl10["test"]])
        self.non_problem_domain_dataset = Subset(imagenet["all"], npd_idx)

        # Prepare target model
        try:
            saved_model = torch.load(target_save_loc)
            self.target_model.load_state_dict(saved_model["state_dict"])
            print("Loaded target model")
        except FileNotFoundError:
            print("Training target model")
            train_model(self.target_model, self.original_domain_dataset, target_save_loc)

        # Prepare PD-OL model
        try:
            saved_model = torch.load(opd_save_loc)
            self.opd_model.load_state_dict(saved_model["state_dict"])
            print("Loaded PD-OL model")
        except FileNotFoundError:
            print("Training PD-OL model")
            train_model(self.opd_model, self.problem_domain_dataset, target_save_loc)

    def test_copycat(self):
        copycat = CopyCat(target_model=self.target_model, opd_model=self.opd_model,
                          copycat_model=self.copycat_model, test_dataset=self.test_dataset,
                          problem_domain_dataset=self.problem_domain_dataset,
                          non_problem_domain_dataset=self.non_problem_domain_dataset,
                          save_loc=save_loc)
        copycat.run()

        self.assertTrue(True)
