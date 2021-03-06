import math
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from pl_bolts.datamodules.sklearn_datamodule import TensorDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomLabelDataset, NoYDataset
from mef.utils.pytorch.functional import get_class_labels
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.settings import AttackSettings


@dataclass
class BlackBoxSettings(AttackSettings):
    iterations: int
    lmbda: float

    def __init__(self, iterations: int, budget: int, lmbda: float):
        self.iterations = iterations
        self.budget = budget
        self.lmbda = lmbda


class BlackBox(Base):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        budget: int = None,
        iterations: int = 6,
        lmbda: float = 0.1,
    ):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = BlackBoxSettings(iterations, budget, lmbda)
        self.trainer_settings._validation = False

    @classmethod
    def _get_attack_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(description="BlackBox attack")
        parser.add_argument(
            "--iterations",
            default=6,
            type=int,
            help="Number of iterations of the attacks (" "Default: 6)",
        )
        parser.add_argument(
            "--budget",
            default=20000,
            type=int,
            help="Size of the budget (Default: 20000). For "
            "the blackbox attack, this is only used as "
            "upper limit.",
        )
        parser.add_argument(
            "--lmbda",
            default=0.1,
            type=float,
            help="Value of lambda in Jacobian augmentation (" "Default: 0.1)",
        )

        return parser

    def _jacobian(self, x: torch.Tensor) -> List[torch.Tensor]:
        list_derivatives = []
        x_var = x.requires_grad_()

        predictions = self._substitute_model(x_var)[0]
        for class_idx in range(self._victim_model.num_classes):
            outputs = predictions[:, class_idx]
            derivative = torch.autograd.grad(
                outputs, x_var, grad_outputs=torch.ones_like(outputs), retain_graph=True
            )[0]
            list_derivatives.append(derivative.cpu().squeeze(dim=0))

        return list_derivatives

    def _jacobian_augmentation(
        self, query_sets: List[Dataset], lmbda: float
    ) -> torch.Tensor:
        thief_dataset = ConcatDataset(query_sets)
        loader = DataLoader(
            dataset=thief_dataset,
            pin_memory=self.base_settings.gpus != 0,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        x_query_set = []
        for x_thief, y_thief in tqdm(
            loader, desc="Jacobian augmentation", total=len(loader)
        ):
            grads = self._jacobian(x_thief)

            y_thief = get_class_labels(y_thief)
            for idx in range(grads[0].shape[0]):
                # Select gradient corresponding to the label predicted by the
                # oracle
                # Need to use max the
                grad = grads[y_thief[idx]][idx]

                # Compute sign matrix
                grad_val = torch.sign(grad)

                # Create new synthetic point in adversary substitute
                # training set
                x_new = x_thief[idx][0] + lmbda * grad_val
                x_query_set.append(x_new.detach().cpu())

        return torch.stack(x_query_set)

    def _check_args(self, sub_data: Dataset, test_set: Dataset) -> None:
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's " "dataset.")
            raise TypeError()
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._thief_dataset = sub_data
        self._test_set = test_set

        return

    def _run(self, sub_data: Dataset, test_set: Dataset) -> None:
        self._check_args(sub_data, test_set)
        self._logger.info("########### Starting BlackBox attack ###########")

        # Get attack's budget
        query_set_size = len(sub_data)
        if self.attack_settings.budget is not None:
            query_set_size = math.floor(
                self.attack_settings.budget / ((2 ** self.attack_settings.iterations) - 1)
            )

        real_budget = (
            query_set_size * (2 ** self.attack_settings.iterations) - query_set_size
        )

        self._logger.info("BlackBox's attack budget: {}".format(real_budget))
        
        if query_set_size != len(sub_data):
            idxs_rest = np.arange(len(self._thief_dataset))
            idxs_initial = np.random.permutation(idxs_rest)[:query_set_size]
            query_data = Subset(self._thief_dataset, idxs_initial)
        else:
            query_data = self._thief_dataset

        y_query_set = self._get_predictions(self._victim_model, query_data)
        query_sets = [CustomLabelDataset(query_data, y_query_set)]
        for it in range(self.attack_settings.iterations):
            self._logger.info("---------- Iteration: {} ----------".format(it + 1))

            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, iteration=it + 1)

            if it < self.attack_settings.iterations - 1:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_query_set = self._jacobian_augmentation(
                    query_sets, self.attack_settings.lmbda
                )

                self._logger.info("Labeling substitute training data")
                # Adversary has access only to labels
                y_query_set = self._get_predictions(
                    self._victim_model, NoYDataset(x_query_set)
                )
                query_sets.append(TensorDataset(x_query_set, y_query_set))

        return
