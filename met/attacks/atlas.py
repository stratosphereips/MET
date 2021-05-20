import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.datamodules.sklearn_datamodule import TensorDataset
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from ..utils.pytorch.datasets import CustomLabelDataset, NoYDataset
from ..utils.pytorch.functional import get_class_labels
from ..utils.pytorch.lightning.module import TrainableModel
from ..utils.pytorch.lightning.trainer import get_trainer_with_settings
from ..utils.settings import AttackSettings
from .base import AttackBase


class UncertaintyPredictor(pl.LightningModule):
    def __init__(self, feature_vec_size):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(feature_vec_size, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    @auto_move_data
    def forward(self, feature_vec):
        return self._model(feature_vec)


@dataclass
class AtlasThiefSettings(AttackSettings):
    iterations: int
    budget: int
    init_seed_size: int
    val_size: int
    k: int

    def __init__(
        self, iterations: int, budget: int, init_seed_size: float, val_size: float
    ):
        self.iterations = iterations
        self.budget = budget

        self.init_seed_size = int(self.budget * init_seed_size)
        self.val_size = int(self.budget * val_size)
        self.k = (self.budget - self.val_size - self.init_seed_size) // self.iterations


class AtlasThief(AttackBase):
    def __init__(
        self,
        victim_model,
        substitute_model,
        iterations=10,
        budget=20000,
        init_seed_size=0.1,
        val_size=0.2,
        *args: Union[int, bool, Path],
        **kwargs: Union[int, bool, Path],
    ):
        super().__init__(victim_model, substitute_model, *args, **kwargs)
        self.attack_settings = AtlasThiefSettings(
            iterations, budget, init_seed_size, val_size
        )

    @classmethod
    def _get_attack_parser(cls):
        parser = argparse.ArgumentParser(description="Atlas attack")
        parser.add_argument(
            "--iterations",
            default=10,
            type=int,
            help="Number of iterations of the attacks (" "Default: " "10)",
        )
        parser.add_argument(
            "--budget",
            default=20000,
            type=int,
            help="Size of the budget (Default: 20000)",
        )

        return parser

    def _get_train_set(self, val_set):
        logits, hidden_layer_output = self._get_predictions(
            self._substitute_model, val_set
        )

        preds = get_class_labels(logits)
        targets = get_class_labels(val_set.targets)

        return (preds == targets).float(), hidden_layer_output

    def _train_new_output_layer(self, train_set):
        trainer_settings = copy.copy(self.trainer_settings)
        trainer_settings.training_epochs = 25
        trainer_settings.validation = False
        trainer, _ = get_trainer_with_settings(
            self.base_settings,
            trainer_settings,
            model_name="correct_model",
            logger=False,
        )

        correct_model = UncertaintyPredictor(train_set[0][0].shape[0])
        loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(correct_model.parameters())
        correct_model = TrainableModel(correct_model, 2, optimizer, loss)

        if self.base_settings.gpu is not None:
            correct_model.cuda(self.base_settings.gpu)

        train_dataloader = DataLoader(
            dataset=train_set,
            pin_memory=True if self.base_settings.gpu is not None else False,
            num_workers=self.base_settings.num_workers,
            shuffle=True,
            batch_size=self.base_settings.batch_size,
        )

        trainer.fit(correct_model, train_dataloader)

        return correct_model

    def _get_atl_sample(self, k, correct_model, hidden_layer_data_rest):
        data = NoYDataset(hidden_layer_data_rest)
        y_preds = self._get_predictions(correct_model, data)
        probs_incorrect = (1 - y_preds).squeeze()

        return torch.argsort(probs_incorrect, dim=-1, descending=True)[:k].numpy()

    def _atlas_strategy(self, data_rest, val_set):
        new_train_labels, new_train_data = self._get_train_set(val_set)
        train_data = [new_train_data]
        train_labels = [new_train_labels]

        _, hidden_layer_outputs_data_rest = self._get_predictions(
            self._substitute_model, data_rest
        )
        hidden_layer_data_rest_all = hidden_layer_outputs_data_rest
        idx_hidden_rest = np.arange(len(hidden_layer_data_rest_all))
        hidden_layer_data_rest = hidden_layer_data_rest_all[idx_hidden_rest]

        selected_points = []
        samples_per_iter = 10
        iterations = range(self.attack_settings.k // samples_per_iter)
        for _ in tqdm(iterations, desc="Selecting best points"):
            train_set = TensorDataset(torch.cat(train_data), torch.cat(train_labels))

            correct_model = self._train_new_output_layer(train_set)
            idxs_best = self._get_atl_sample(
                samples_per_iter, correct_model, hidden_layer_data_rest
            )

            selected_points.append(idxs_best)
            train_data.append(hidden_layer_data_rest[idxs_best])
            train_labels.append(torch.ones((samples_per_iter, 1)).float())

            idx_hidden_rest = np.setdiff1d(idx_hidden_rest, idxs_best)
            hidden_layer_data_rest = hidden_layer_data_rest_all[idx_hidden_rest]

        return np.concatenate(selected_points)

    def _check_args(self, sub_data: Type[Dataset], test_set: Type[Dataset]):
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's " "dataset.")
            raise TypeError()
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._adversary_dataset = sub_data
        self._test_set = test_set

        return

    def _run(self, sub_data: Type[Dataset], test_set: Type[Dataset]):
        self._check_args(sub_data, test_set)
        self._logger.info("########### Starting AtlasThief attack ###########")
        # Get budget of the attack
        self._logger.info(
            "AtlasThief's attack budget: {}".format(self.attack_settings.budget)
        )

        idxs_rest = np.arange(len(self._adversary_dataset))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idxs_val = np.random.permutation(idxs_rest)[: self.attack_settings.val_size]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_val)
        val_set = Subset(self._adversary_dataset, idxs_val)
        y_val = self._get_predictions(self._victim_model, val_set)
        val_set = CustomLabelDataset(val_set, y_val)

        val_label_counts = dict(list(enumerate([0] * self._victim_model.num_classes)))
        if y_val.size()[-1] == 1:
            for class_id in torch.round(y_val):
                val_label_counts[class_id.item()] += 1
        else:
            for class_id in torch.argmax(y_val, dim=-1):
                val_label_counts[class_id.item()] += 1

        self._logger.info(
            "Validation dataset labels distribution: {}".format(val_label_counts)
        )

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        query_sets = []

        idxs_query = np.random.permutation(idxs_rest)[
            : self.attack_settings.init_seed_size
        ]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
        query_set = Subset(self._adversary_dataset, idxs_query)
        y_query = self._get_predictions(self._victim_model, query_set)
        query_sets.append(CustomLabelDataset(query_set, y_query))

        # Get victim model metrics on test set
        self._logger.info("Getting victim model's metrics for test set")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)

        # Save substitute model state_dict for retraining from scratch
        sub_orig_state_dict = self._substitute_model.state_dict()
        optim_orig_state_dict = self._substitute_model.optimizer.state_dict()

        for it in range(self.attack_settings.iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it + 1))

            # Reset substitute model and optimizer
            self._substitute_model.load_state_dict(sub_orig_state_dict)
            self._substitute_model.optimizer.load_state_dict(optim_orig_state_dict)

            # Step 3: The substitute model is trained with union of all the
            # labeled queried sets
            self._logger.info("Training substitute model with the query dataset")
            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, val_set, it + 1)

            if (it + 1) == (self.attack_settings.iterations + 1):
                break

            # Get metrics from victim model and substitute model
            self._logger.info("Getting substitute model's metrics for test " "set")
            sub_test_acc = self._test_model(self._substitute_model, self._test_set)
            self._logger.info("Test set metrics")
            self._logger.info("Victim model Accuracy: {:.1f}%".format(vict_test_acc))
            self._logger.info("Substitute model Accuracy: {:.1f}%".format(sub_test_acc))
            self._get_aggreement_score()

            # Step 4: An Atlas subset selection strategy is used
            # to select set of k samples
            self._logger.info(
                "Selecting {} samples from the remaining thief dataset".format(
                    self.attack_settings.k
                )
            )
            data_rest = Subset(self._adversary_dataset, idxs_rest)
            idxs_query = self._atlas_strategy(data_rest, val_set)
            idxs_query = idxs_rest[idxs_query]
            idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
            query_set = Subset(self._adversary_dataset, idxs_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info(
                "Getting predictions for the current query set " "from the victim model"
            )
            y_query = self._get_predictions(self._victim_model, query_set)
            query_sets.append(CustomLabelDataset(query_set, y_query))

        return
