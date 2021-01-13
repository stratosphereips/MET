import argparse
from dataclasses import dataclass
from typing import Type

import torch
from pl_bolts.datamodules.sklearn_datamodule import TensorDataset
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import NoYDataset
from mef.utils.pytorch.functional import get_class_labels
from mef.utils.settings import AttackSettings


@dataclass
class BlackBoxSettings(AttackSettings):
    iterations: int
    lmbda: float

    def __init__(self,
                 iterations: int,
                 lmbda: float):
        self.iterations = iterations
        self.lmbda = lmbda


class BlackBox(Base):

    def __init__(self,
                 victim_model,
                 substitute_model,
                 iterations=6,
                 lmbda=0.1):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = BlackBoxSettings(iterations, lmbda)
        self.trainer_settings._validation = False

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="BlackBox attack")
        parser.add_argument("--iterations", default=6, type=int,
                            help="Number of iterations of the attacks ("
                                 "Default: 6)")
        parser.add_argument("--lmbda", default=0.1, type=float,
                            help="Value of lambda in Jacobian augmentation ("
                                 "Default: 0.1)")
        parser.add_argument("--training_epochs", default=10, type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 10)")

        cls._add_base_args(parser)

        return parser

    def _jacobian(self, x):
        list_derivatives = []
        x_var = x.requires_grad_()

        predictions = self._substitute_model(x_var)[0]
        for class_idx in range(self._victim_model.num_classes):
            outputs = predictions[:, class_idx]
            derivative = torch.autograd.grad(
                    outputs,
                    x_var,
                    grad_outputs=torch.ones_like(outputs),
                    retain_graph=True)[0]
            list_derivatives.append(derivative.cpu().squeeze(dim=0))

        return list_derivatives

    def _jacobian_augmentation(self, query_sets, lmbda):
        thief_dataset = ConcatDataset(query_sets)
        loader = DataLoader(dataset=thief_dataset,
                            pin_memory=self.base_settings.gpus != 0,
                            num_workers=self.base_settings.num_workers,
                            batch_size=self.base_settings.batch_size)

        x_query_set = []
        for x_thief, y_thief in tqdm(loader, desc="Jacobian augmentation",
                                     total=len(loader)):
            grads = self._jacobian(x_thief)

            for idx in range(grads[0].shape[0]):
                # Select gradient corresponding to the label predicted by the
                # oracle
                grad = grads[y_thief[idx]][idx]

                # Compute sign matrix
                grad_val = torch.sign(grad)

                # Create new synthetic point in adversary substitute
                # training set
                x_new = x_thief[idx][0] + lmbda * grad_val
                x_query_set.append(x_new.detach().cpu())

        return torch.stack(x_query_set)

    def _check_args(self,
                    sub_data: Type[Dataset],
                    test_set: Type[Dataset]):
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's "
                               "dataset.")
            raise TypeError()
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._thief_dataset = sub_data
        self._test_set = test_set

        return

    def _run(self,
             sub_data: Type[Dataset],
             test_set: Type[Dataset]):
        self._check_args(sub_data, test_set)
        self._logger.info("########### Starting BlackBox attack ###########")

        # Get attack's budget
        budget = len(self._thief_dataset) * \
                 (2 ** self.attack_settings.iterations) - \
                 len(self._thief_dataset)
        self._logger.info("BlackBox's attack budget: {}".format(budget))

        query_sets = [self._thief_dataset]
        for it in range(self.attack_settings.iterations):
            self._logger.info("---------- Iteration: {} ----------".format(
                    it + 1))

            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, iteration=it + 1)

            if it < self.attack_settings.iterations - 1:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_query_set = self._jacobian_augmentation(query_sets,
                                                          self.attack_settings.lmbda)

                self._logger.info("Labeling substitute training data")
                # Adversary has access only to labels
                y_query_set = self._get_predictions(self._victim_model,
                                                    NoYDataset(x_query_set))
                y_query_set = get_class_labels(y_query_set)

                query_sets.append(TensorDataset(x_query_set,
                                                y_query_set.numpy()))

        return
