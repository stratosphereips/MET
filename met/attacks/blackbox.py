from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import foolbox as fb
import numpy as np
import torch
from pl_bolts.datamodules.sklearn_datamodule import TensorDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from ..utils.pytorch.datasets import CustomLabelDataset, NoYDataset
from ..utils.pytorch.functional import get_class_labels
from ..utils.pytorch.lightning.module import TrainableModel, VictimModel
from ..utils.settings import AttackSettings
from .base import AttackBase


@dataclass
class BlackBoxSettings(AttackSettings):
    iterations: int
    bounds: Tuple[float, float]
    lmbda: float
    _adversary_strategy: str

    def __init__(
        self,
        iterations: int,
        bounds: Tuple[float, float],
        lmbda: float,
        # adversary_strategy: str,
    ):
        self.iterations = iterations
        self.bounds = bounds
        self.lmbda = lmbda
        self._adversary_strategy = "N FGSM"

        # # Check configuration
        # if self.adversary_strategy not in [
        #     "N FGSM",
        #     "T-RND FGSM",
        #     "N I-FGSM",
        #     "T-RND I-FGSM",
        # ]:
        #     raise ValueError(
        #         "BlackBox's adversary strategy must be one of {N FGSM, T-RND FGSM, N I-FGSM, T-RND I-FGSM}"
        #     )


class BlackBox(AttackBase):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        bounds: Tuple[float, float] = (0, 1),
        iterations: int = 6,
        lmbda: float = 0.1,
        # adversary_strategy: str = "N FGSM",
        *args: Union[int, bool, Path],
        **kwargs: Union[int, bool, Path],
    ):
        """Implementation of BlackBox attack from .

        Args:
            victim_model (VictimModel): Victim model, which is the target of the attack, wrapped inside the VictimModel class.
            substitute_model (TrainableModel): Substitue model, which the attack will train.
            bounds (Tuple[float, float], optional): Bounds for Deepfool attack represented as tuple (min, max). Defaults to (0, 1).
            iterations (int, optional): For how many iterations should the attack run. Defaults to 6.
            lmbda (float, optional): Lambda that should be used in the Jacobian augmentation. Defaults to 0.1.
            training_epochs (int, optional): Number of training epochs for which the substitute model should be trained. Defaults to 1000.
            patience (int, optional): Patience for the early stopping during training of substiute model. If specified early stopping will be used. Defaults to None.
            evaluation_frequency (int, optional): Evalution frequency if validation set is available during training of a substitute model. Some attacks will automatically create validation set from adversary dataset if the user did not specify it himself. Defaults to None.
            precision (int, optional): Number precision that should be used. Currently only used in the pytorch-lightning trainer. Defaults to 32.
            use_accuracy (bool, optional): Whether to use accuracy during validation for checkpointing or F1-score, which is used by default. Defaults to False.
            save_loc (Path, optional): Location where log and other files created during the attack should be saved. Defaults to Path("./cache/").
            gpu (int, optional): Id of the gpu that should be used for the training. Defaults to None.
            num_workers (int, optional): Number of workers that should be used for data loaders. Defaults to 1.
            batch_size (int, optional): Batch size that should be used throughout the attack. Defaults to 32.
            seed (int, optional): Seed that should be used to initialize random generators, to help with reproducibility of results. Defaults to None.
            deterministic (bool, optional): Whether training should tried to be deterministic. Defaults to False.
            debug (bool, optional): Adds additional details to the log file and also performs all testing, training with only one batch. Defaults to False.
        """
        super().__init__(victim_model, substitute_model, *args, **kwargs)
        self.attack_settings = BlackBoxSettings(iterations, bounds, lmbda)

    @classmethod
    def _get_attack_parser(
        cls, parser: Optional[ArgumentParser] = None
    ) -> ArgumentParser:
        parser = (
            ArgumentParser(description="BlackBox attack") if parser is None else parser
        )
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
            help="Value of lambda in Jacobian augmentation, it represents the epsilon value in the FGSM method. (Default: 0.1)",
        )

        return parser

    def _select_adversary_attack(self) -> fb.attacks.Attack:
        if self.attack_settings._adversary_strategy == "N FGSM":
            return fb.attacks.FGSM()
        elif self.attack_settings._adversary_strategy == "T-RND FGSM":
            return fb.attacks.LinfBasicIterativeAttack(rel_stepsize=1, steps=1)
        elif self.attack_settings._adversary_strategy in ["N I-FGSM", "T-RND I-FGSM"]:
            return fb.attacks.LinfBasicIterativeAttack(
                steps=11, abs_stepsize=self.attack_settings.lmbda / 11
            )

    def _create_synthetic_samples(self, query_sets: List[Dataset]) -> torch.Tensor:
        thief_dataset = ConcatDataset(query_sets)
        loader = DataLoader(
            dataset=thief_dataset,
            pin_memory=True if self.base_settings.gpu is not None else False,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )
        device = (
            f"cuda:{self.base_settings.gpu}"
            if self.base_settings.gpu is not None
            else "cpu"
        )
        model = fb.PyTorchModel(
            self._substitute_model.model,
            bounds=self.attack_settings.bounds,
            device=device,
        )

        attack = self._select_adversary_attack()

        x_query_set = []
        for x_thief, y_thief in tqdm(
            loader, desc="Generating synthetic samples", total=len(loader)
        ):
            if self.base_settings.gpu is not None:
                x_thief = x_thief.cuda(f"cuda:{self.base_settings.gpu}")
                y_thief = y_thief.cuda(f"cuda:{self.base_settings.gpu}")

            labels = get_class_labels(y_thief)
            criterion = fb.criteria.Misclassification(labels)
            if "T-RND" in self.attack_settings._adversary_strategy:
                targets = []
                for label in labels:
                    classes = np.arange(self._victim_model.num_classes, dtype=np.int64)
                    classes = np.delete(classes, label.cpu().item())
                    targets.append(np.random.choice(classes))
                targets = torch.tensor(targets)
                if self.base_settings.gpu is not None:
                    targets = targets.cuda(f"cuda:{self.base_settings.gpu}")
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
