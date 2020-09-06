from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.utils.data import Dataset, Subset, DataLoader

data_root = "../data"
adult_root = "E:/Datasets/Adult"
train_epochs = 20
device = "cuda"


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class AdultDataset(Dataset):
    """Adult dataset"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the adult.data csv file.
        """
        self._adult_data = pd.read_csv(csv_file)
        self._x = None
        self._y = None
        self._prepare_data()

    def _prepare_data(self):
        # replacing some special character columns names with proper names
        self._adult_data.rename(
            columns={"capital-gain": "capital gain", "capital-loss": "capital loss",
                     "native-country": "country", "hours-per-week": "hours per week",
                     "marital-status": "marital"}, inplace=True)

        # code will replace the special character to nan and then drop the columns
        self._adult_data["country"] = self._adult_data["country"].replace('?', np.nan)
        self._adult_data["workclass"] = self._adult_data["workclass"].replace('?', np.nan)
        self._adult_data["occupation"] = self._adult_data["occupation"].replace('?', np.nan)
        # dropping the NaN rows now
        self._adult_data.dropna(how="any", inplace=True)

        # dropping based on uniquness of data from the dataset
        self._adult_data.drop(
            ["educational-num", "age", "hours per week", "fnlwgt", "capital gain", "capital loss",
             "country"], axis=1, inplace=True)

        # mapping the data into numerical data using map function
        self._adult_data["income"] = self._adult_data["income"].map({"<=50K": 0, ">50K": 1}) \
            .astype(float)
        # gender
        self._adult_data["gender"] = self._adult_data["gender"].map({"Male": 0, "Female": 1}) \
            .astype(float)
        # race
        self._adult_data["race"] = self._adult_data["race"].map(
            {"Black": 0, "Asian-Pac-Islander": 1, "Other": 2, "White": 3,
             "Amer-Indian-Eskimo": 4}).astype(float)
        # marital
        self._adult_data["marital"] = self._adult_data["marital"].map(
            {"Married-spouse-absent": 0, "Widowed": 1, "Married-civ-spouse": 2, "Separated": 3,
             "Divorced": 4, "Never-married": 5, "Married-AF-spouse": 6}).astype(float)
        # workclass
        self._adult_data["workclass"] = self._adult_data["workclass"].map(
            {"Self-emp-inc": 0, "State-gov": 1, "Federal-gov": 2, "Without-pay": 3, "Local-gov": 4,
             "Private": 5, "Self-emp-not-inc": 6}).astype(float)
        # education
        self._adult_data["education"] = self._adult_data["education"].map(
            {"Some-college": 0, "Preschool": 1, "5th-6th": 2, "HS-grad": 3, "Masters": 4, "12th": 5,
             "7th-8th": 6, "Prof-school": 7, "1st-4th": 8, "Assoc-acdm": 9, "Doctorate": 10,
             "11th": 11, "Bachelors": 12, "10th": 13, "Assoc-voc": 14, "9th": 15}).astype(float)
        # occupation
        self._adult_data["occupation"] = self._adult_data["occupation"].map(
            {"Farming-fishing": 1, "Tech-support": 2, "Adm-clerical": 3, "Handlers-cleaners": 4,
             "Prof-specialty": 5, "Machine-op-inspct": 6, "Exec-managerial": 7,
             "Priv-house-serv": 8, "Craft-repair": 9, "Sales": 10, "Transport-moving": 11,
             "Armed-Forces": 12, "Other-service": 13, "Protective-serv": 14}).astype(float)
        # relationship
        self._adult_data["relationship"] = self._adult_data["relationship"].map(
            {"Not-in-family": 0, "Wife": 1, "Other-relative": 2, "Unmarried": 3, "Husband": 4,
             "Own-child": 5}).astype(float)

        self._x = self._adult_data.drop(["income"], axis=1)
        self._y = self._adult_data["income"]

    def __len__(self):
        return len(self._adult_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        attributes = self._x.iloc[idx, :].values
        income = self._y.iloc[idx]

        if hasattr(income, "values"):
            income = income.values

        return torch.from_numpy(attributes).float(), torch.tensor(income).float().view(1)


class TestCustomDataset(TestCase):
    def setUp(self) -> None:
        self.adult_data = AdultDataset(adult_root + "/adult.csv")
        self.num_features = len(self.adult_data._x.columns)
        self.log_reg = LogisticRegression(num_features=self.num_features)

        if device == "cuda":
            self.log_reg.cuda()

    def test_customdataset(self):
        # Prepare data
        available_samples = set(range(len(self.adult_data)))
        idx = np.random.choice(np.arange(len(available_samples)),
                               size=int(round((len(self.adult_data) * 0.7), 0)), replace=False)
        available_samples -= set(idx)

        train_split = Subset(self.adult_data, idx)
        self.test_split = Subset(self.adult_data, sorted(available_samples))

        optimizer = torch.optim.SGD(self.log_reg.parameters(), lr=0.01)
        loss_function = torch.nn.BCELoss()
        trainer = create_supervised_trainer(self.log_reg, loss_fn=loss_function,
                                            optimizer=optimizer, device=device)

        ProgressBar().attach(trainer)

        # Train model
        print("Training model")
        train_loader = DataLoader(train_split, batch_size=64, num_workers=4, shuffle=True,
                                  pin_memory=True)
        trainer.run(train_loader, max_epochs=train_epochs)

        val_metrics = {
            "accuracy": Accuracy(),
            "bce": Loss(loss_function)
        }
        evaluator = create_supervised_evaluator(self.log_reg, metrics=val_metrics,
                                                device=device,
                                                output_transform=lambda x, y, y_pred:
                                                (y_pred.round_(), y))
        test_loader = DataLoader(self.test_split, batch_size=256, num_workers=4,
                                 pin_memory=True)
        evaluator.run(test_loader)

        metrics = evaluator.state.metrics
        print("Training Result Avg accuracy: {:.2f} Avg loss: {:.2f}".format(metrics["accuracy"],
                                                                             metrics["bce"]))

        self.assertTrue(True)
