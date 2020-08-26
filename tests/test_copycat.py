from unittest import TestCase

import numpy as np
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import CIFAR10, STL10, ImageNet
from torchvision.transforms import transforms

import mef
from mef.attacks.copycat import CopyCat
from mef.models.vision.vgg import Vgg
from mef.utils.details import ModelDetails
from mef.utils.pytorch.training import ModelTraining

config_file = "../config.yaml"
data_root = "../data"
imagenet_root = "E:/Datasets/ImageNet2012"
npd_size = 5000


class TestCopyCat(TestCase):
    def setUp(self) -> None:
        mef.Test(config_file)

        model_details = ModelDetails(net=dict(name="vgg_16",
                                              act="relu",
                                              drop="none",
                                              pool="max_2",
                                              ks=3,
                                              n_conv=13,
                                              n_fc=3),
                                     opt=dict(
                                         name="SGD",
                                         batch_size=64,
                                         epochs=1,
                                         momentum=0.5,
                                         lr=0.1),
                                     loss=dict(name="cross_entropy"))

        self.target_model = Vgg(input_dimensions=(3, 32, 32), n_classes=9,
                                model_details=model_details)
        self.opd_model = Vgg(input_dimensions=(3, 32, 32), n_classes=9, model_details=model_details)
        self.copycat_model = Vgg(input_dimensions=(3, 32, 32), n_classes=9,
                                 model_details=model_details)

        # Prepare data
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

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]

        cifar10 = dict()
        cifar10["train"] = CIFAR10(root=data_root, download=True,
                                   transform=transforms.Compose(transform))

        cifar10["test"] = CIFAR10(root=data_root, train=False, download=True,
                                  transform=transforms.Compose(transform))

        transform = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean,
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

        transform = [transforms.Resize((32, 32)), transforms.ToTensor()]
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

        # Prepare target and opd model
        ModelTraining.train_model(self.target_model, training_data=self.original_domain_dataset)
        ModelTraining.train_model(self.opd_model, training_data=self.problem_domain_dataset)

    def test_copycat(self):
        copycat = CopyCat(target_model=self.target_model, opd_model=self.opd_model,
                          copycat_model=self.copycat_model, test_dataset=self.test_dataset,
                          problem_domain_dataset=self.problem_domain_dataset,
                          non_problem_domain_dataset=self.non_problem_domain_dataset)
        copycat.run()

        self.assertTrue(True)
