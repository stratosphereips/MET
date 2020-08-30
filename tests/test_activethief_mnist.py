from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import MNIST, ImageNet
from torchvision.transforms import transforms

import mef
from mef.attacks.activethief import ActiveThief
from mef.models.vision.mnet import Mnet
from mef.utils.details import ModelDetails
from mef.utils.ios import mkdir_if_missing

config_file = "../config.yaml"
data_root = "../data"
imagenet_root = "E:/Datasets/ImageNet2012"
save_loc = "./cache/activethief/MNIST"
secret_savel_loc = save_loc + "/secret_model.pt"
thief_dataset_size = 120000
validation_dataset_size = 4000
train_epochs = 10
device = "cuda"


class TestActiveThief(TestCase):
    def setUp(self) -> None:
        mkdir_if_missing(save_loc)
        mef.Test(config_file)

        secret_details = ModelDetails(net=dict(name="mnet",
                                               act="relu",
                                               drop="none",
                                               pool="max_2",
                                               ks=3,
                                               n_conv=3,
                                               n_fc=3), )
        substitute_details = ModelDetails(net=dict(name="mnet",
                                                   act="relu",
                                                   drop="none",
                                                   pool="max_2",
                                                   ks=3,
                                                   n_conv=3,
                                                   n_fc=3))
        self.secret_model = Mnet(input_dimensions=(1, 28, 28), num_classes=10,
                                 model_details=secret_details)
        self.substitute_model = Mnet(input_dimensions=(1, 28, 28), num_classes=10,
                                     model_details=substitute_details)

        if device == "cuda":
            self.secret_model.cuda()
            self.substitute_model.cuda()

        # Prepare data
        print("Preparing data")
        transform = [transforms.ToTensor()]
        mnist = dict()
        mnist["train"] = MNIST(root=data_root, download=True,
                               transform=transforms.Compose(transform))

        mnist["test"] = MNIST(root=data_root, train=False, download=True,
                              transform=transforms.Compose(transform))
        self.test_dataset = mnist["test"]

        transform = [transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor()]
        imagenet = dict()
        imagenet["train"] = ImageNet(root=imagenet_root, transform=transforms.Compose(transform))
        imagenet["test"] = ImageNet(root=imagenet_root, split="val",
                                    transform=transforms.Compose(transform))
        imagenet["all"] = ConcatDataset(imagenet.values())

        available_samples = set(range(len(imagenet["all"])))
        idx = np.random.choice(np.arange(len(imagenet["all"])), size=thief_dataset_size,
                               replace=False)
        available_samples -= set(idx)

        self.thief_dataset = Subset(imagenet["all"], idx)

        idx = np.random.choice(sorted(list(available_samples)), size=validation_dataset_size,
                               replace=False)
        self.validation_dataset = Subset(imagenet["all"], idx)

        # Train secret model
        try:
            saved_model = torch.load(secret_savel_loc)
            self.secret_model.load_state_dict(saved_model["state_dict"])
            print("Loaded target model")
        except FileNotFoundError:
            # Prepare secret model
            print("Training secret model")
            train_loader = DataLoader(dataset=mnist["train"], batch_size=128, shuffle=True,
                                      num_workers=4, pin_memory=True)
            optimizer = torch.optim.Adam(self.secret_model.parameters())
            loss_function = F.cross_entropy
            trainer = create_supervised_trainer(self.secret_model, optimizer, loss_function,
                                                device=device)
            ProgressBar().attach(trainer)
            trainer.run(train_loader, max_epochs=train_epochs)

            torch.save(dict(state_dict=self.secret_model.state_dict()), secret_savel_loc)

    def test_activethief(self):
        active_thief = ActiveThief(secret_model=self.secret_model,
                                   substitute_model=self.substitute_model,
                                   test_dataset=self.test_dataset,
                                   thief_dataset=self.thief_dataset,
                                   validation_dataset=self.validation_dataset)
        active_thief.run()

        self.assertTrue(True)
