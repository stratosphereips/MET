from unittest import TestCase

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import mef
from mef.models.vision.simplenet import SimpleNet
from mef.utils.details import ModelDetails
from mef.utils.pytorch.training import ModelTraining


class TestModelTraining(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mef.Test("../config.yaml")

        model_details = ModelDetails(net=dict(name="simplenet",
                                              act="relu",
                                              drop="normal",
                                              pool="max_2",
                                              ks=3,
                                              n_conv=13,
                                              n_fc=1),
                                     opt=dict(name="SGD",
                                              batch_size=64,
                                              epochs=1,
                                              momentum=0.5,
                                              lr=0.1),
                                     loss=dict(name="cross_entropy"))
        cls.model = SimpleNet((1, 28, 28), 10, model_details)

    def test1_train_model(self):
        trainining_dataset = MNIST("./data", download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))

        accuracy, f1score = ModelTraining.train_model(model=self.model,
                                                      training_data=trainining_dataset)

        self.assertIsNone(accuracy)
        self.assertIsNone(f1score)

    def test2_evaluate_model(self):
        evaluation_dataset = MNIST("./data", download=True, train=False,
                                   transform=transforms.Compose([transforms.ToTensor()]))

        accuracy, f1score = ModelTraining.evaluate_model(self.model,
                                                         evaluation_data=evaluation_dataset)

        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(f1score)
