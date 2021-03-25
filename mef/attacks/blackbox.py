from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Tuple

import foolbox as fb
import numpy as np
import torch
from pl_bolts.datamodules.sklearn_datamodule import TensorDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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

    def __init__(
        self,
        iterations: int,
        bounds: Tuple[float, float],
        lmbda: float,
        adversary_strategy: str,
    ):
        self.iterations = iterations
        self.bounds = bounds
        self.lmbda = lmbda
        self.adversary_strategy = adversary_strategy

        # Check configuration
        if self.adversary_strategy not in [
            "N FGSM",
            "T-RND FGSM",
            "N I-FGSM",
            "T-RND I-FGSM",
        ]:
            raise ValueError(
                "BlackBox's adversary strategy must be one of {"
                "N FGSM, T-RND FGSM, N I-FGSM, T-RND I-FGSM}"
            )


class BlackBox(Base):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        bounds: Tuple[float, float] = (0, 1),
        iterations: int = 6,
        lmbda: float = 0.1,
        adversary_strategy: str = "N FGSM",
    ):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = BlackBoxSettings(
            iterations, bounds, lmbda, adversary_strategy
        )
        self.trainer_settings._validation = False

    @classmethod
    def _get_attack_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(description="BlackBox attack")
        parser.add_argument(
            "--iterations",
            default=6,
            type=int,
            help="Number of iterations of the attacks (Default: 6)",
        )
        parser.add_argument(
            "--lmbda",
            default=0.1,
            type=float,
            help="Value of lambda in Jacobian augmentation (Default: 0.1)",
        )

        return parser

    def _select_adversary_attack(self):
        if self.attack_settings.adversary_strategy == "N FGSM":
            return fb.attacks.FGSM()
        elif self.attack_settings.adversary_strategy == "T-RND FGSM":
            return fb.attacks.LinfBasicIterativeAttack(rel_stepsize=1, steps=1)
        elif self.attack_settings.adversary_strategy in ["N I-FGSM", "T-RND I-FGSM"]:
            return fb.attacks.LinfBasicIterativeAttack(
                steps=11, abs_stepsize=self.attack_settings.lmbda/11
            )

    def _create_synthetic_samples(self, query_sets: List[Dataset]):
        thief_dataset = ConcatDataset(query_sets)
        loader = DataLoader(
            dataset=thief_dataset,
            pin_memory=self.base_settings.gpus != 0,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )
        model = fb.PyTorchModel(
            self._substitute_model.model, bounds=self.attack_settings.bounds
        )
        attack = self._select_adversary_attack()

        x_query_set = []
        for x_thief, y_thief in tqdm(
            loader, desc="Generating synthetic samples", total=len(loader)
        ):
            if self.base_settings.gpus:
                x_thief = x_thief.cuda()
                y_thief = y_thief.cuda()

            labels = get_class_labels(y_thief)
            criterion = fb.criteria.Misclassification(labels)
            if "T-RND" in self.attack_settings.adversary_strategy:
                targets = []
                for label in labels:
                    classes = np.arange(self._victim_model.num_classes, dtype=np.int64)
                    classes = np.delete(classes, label.cpu().item())
                    targets.append(np.random.choice(classes))
                targets = torch.tensor(targets)
                if self.base_settings.gpus:
                    targets = targets.cuda()
                criterion = fb.criteria.TargetedMisclassification(targets)

            x_synthetic_new, _, _ = attack(
                model,
                x_thief,
                criterion=criterion,
                epsilons=self.attack_settings.lmbda,
            )
            x_query_set.append(x_synthetic_new.detach().cpu())

        return torch.cat(x_query_set)

    def _check_args(self, sub_data: Dataset, test_set: Dataset) -> None:
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's dataset.")
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
        real_budget = len(sub_data) * (2 ** self.attack_settings.iterations)
        self._logger.info("BlackBox's attack budget: {}".format(real_budget))

        query_data = self._thief_dataset
        y_query_set = self._get_predictions(self._victim_model, query_data)
        query_sets = [CustomLabelDataset(query_data, y_query_set)]
        for it in range(self.attack_settings.iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it + 1))

            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, iteration=it + 1)

            if it < self.attack_settings.iterations:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_query_set = self._create_synthetic_samples(query_sets)

                self._logger.info("Labeling substitute training data")
                # Adversary has access only to labels
                y_query_set = self._get_predictions(
                    self._victim_model, NoYDataset(x_query_set)
                )
                query_sets.append(TensorDataset(x_query_set, y_query_set))

        return
