import argparse
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from mef.attacks.base import AttackSettings, Base
from mef.utils.pytorch.datasets import CustomDataset, CustomLabelDataset, \
    MefDataset, NoYDataset
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer


class UncertaintyPredictor(pl.LightningModule):

    def __init__(self, feature_vec_size):
        super().__init__()
        self._model = nn.Sequential(nn.Linear(feature_vec_size, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 2))

    @auto_move_data
    def forward(self, feature_vec):
        return self._model(feature_vec)


@dataclass
class ActiveThiefSettings(AttackSettings):
    iterations: int
    output_type: str
    budget: int
    init_seed_size: int
    val_size: int
    k: int

    def __init__(self,
                 iterations: int,
                 output_type: str,
                 budget: int):
        self.iterations = iterations
        self.output_type = output_type.lower()
        self.budget = budget

        self.init_seed_size = int(self.budget * 0.1)
        self.val_size = int(self.budget * 0.2)
        self.k = (self.budget - self.val_size - self.init_seed_size) // \
                 self.iterations


class AtlasThief(Base):

    def __init__(self,
                 victim_model,
                 substitute_model,
                 num_classes,
                 iterations=10,
                 output_type="softmax",
                 budget=20000):
        optimizer = torch.optim.Adam(substitute_model.parameters(),
                                     weight_decay=1e-3)
        loss = soft_cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss)
        self.attack_settings = ActiveThiefSettings(iterations, output_type,
                                                   budget)
        self.data_settings._num_classes = num_classes

    @classmethod
    def get_attack_args(self):
        parser = argparse.ArgumentParser(description="Atlas attack")
        parser.add_argument("--iterations", default=10, type=int,
                            help="Number of iterations of the attacks ("
                                 "Default: "
                                 "10)")
        parser.add_argument("--output_type", default="softmax", type=str,
                            help="Type of output from victim model {softmax, "
                                 "logits, one_hot} (Default: softmax)")
        parser.add_argument("--budget", default=20000, type=int,
                            help="Size of the budget (Default: 20000)")
        parser.add_argument("--training_epochs", default=1000,
                            type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 100)")
        parser.add_argument("--patience", default=100, type=int,
                            help="Number of epochs without improvement for "
                                 "early stop (Default: 100)")
        parser.add_argument("--evaluation_frequency", default=1, type=int,
                            help="Epochs interval of validation (Default: 1)")

        self._add_base_args(parser)

        return parser

    def _get_train_set(self,
                       val_set):
        preds, hidden_layer_output = self._get_predictions(
                self._substitute_model, val_set, return_all_layers=True)

        preds = preds.argmax(dim=1)
        targets = val_set.targets.argmax(dim=1)

        return (preds == targets).long(), hidden_layer_output

    def _train_new_output_layer(self,
                                train_set):
        base_settings = self.base_settings.copy()
        trainer_settings = self.trainer_settings.copy()
        trainer_settings.training_epochs = 25
        trainer_settings.validation = False
        trainer = get_trainer(base_settings, trainer_settings, "correct_model")

        correct_model = UncertaintyPredictor(train_set[0][0].shape[0])
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(correct_model.parameters(), lr=0.01,
                                    momentum=0.5)
        mef_model = MefModule(correct_model, 2, optimizer, loss)

        train_set = MefDataset(train_set, self.data_settings.batch_size)
        train_loader = train_set.train_dataloader()
        trainer.fit(mef_model, train_loader)

        return correct_model

    def _get_atl_sample(self,
                        k,
                        correct_model,
                        hidden_layer_data_rest):
        y_preds = self._get_predictions(correct_model, hidden_layer_data_rest)
        probs_incorrect = y_preds.data[:, 0]

        return probs_incorrect.topk(k).indices.numpy()

    def _atlas_strategy(self,
                        data_rest,
                        val_set):
        new_train_labels, new_train_data = self._get_train_set(val_set)
        train_data = [new_train_data]
        train_labels = [new_train_labels]

        _, hidden_layer_outputs_data_rest = self._get_predictions(
                self._substitute_model, data_rest, return_all_layers=True)
        hidden_layer_data_rest_all = NoYDataset(hidden_layer_outputs_data_rest)

        idx_hidden_rest = np.arange(len(hidden_layer_data_rest_all))
        hidden_layer_data_rest = Subset(hidden_layer_data_rest_all,
                                        idx_hidden_rest)

        selected_points = []
        samples_per_iter = 10
        iterations = range(self.attack_settings.k // samples_per_iter)
        for _ in tqdm(iterations, desc="Selecting best points"):
            train_set = CustomDataset(torch.cat(train_data),
                                      torch.cat(train_labels))

            correct_model = self._train_new_output_layer(train_set)
            idxs_best = self._get_atl_sample(samples_per_iter, correct_model,
                                             hidden_layer_data_rest)

            selected_points.append(idxs_best)
            train_data.append(hidden_layer_data_rest[idxs_best][0])
            train_labels.append(torch.ones(samples_per_iter).long())

            idx_hidden_rest = np.setdiff1d(idx_hidden_rest, idxs_best)
            hidden_layer_data_rest = Subset(hidden_layer_data_rest_all,
                                            idx_hidden_rest)

        return np.concatenate(selected_points)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info(
                "########### Starting AtlasThief attack ###########")
        # Get budget of the attack
        self._logger.info("AtlasThief's attack budget: {}".format(
                self.attack_settings.budget))

        idxs_rest = np.arange(len(self._thief_dataset))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idxs_val = np.random.permutation(idxs_rest)[
                   :self.attack_settings.val_size]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_val)
        val_set = Subset(self._thief_dataset, idxs_val)
        y_val = self._get_predictions(self._victim_model, val_set,
                                      self.attack_settings.output_type)
        val_set = CustomLabelDataset(val_set, y_val)

        val_label_counts = dict(
                list(enumerate([0] * self.data_settings._num_classes)))
        for class_id in torch.argmax(y_val, dim=1):
            val_label_counts[class_id.item()] += 1

        self._logger.info("Validation dataset labels distribution: {}".format(
                val_label_counts))

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        query_sets = []

        idxs_query = np.random.permutation(idxs_rest)[
                     :self.attack_settings.init_seed_size]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
        query_set = Subset(self._thief_dataset, idxs_query)
        y_query = self._get_predictions(self._victim_model, query_set,
                                        self.attack_settings.output_type)
        query_sets.append(CustomLabelDataset(query_set, y_query))

        # Get victim model predicted labels for test set
        self._logger.info("Getting victim model's labels for test set")
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=1)
        self._logger.info(
                "Number of test samples: {}".format(len(vict_test_labels)))

        # Get victim model metrics on test set
        self._logger.info("Getting victim model's metrics for test set")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)

        # Save substitute model state_dict for retraining from scratch
        sub_orig_state_dict = self._substitute_model.state_dict()
        optim_orig_state_dict = self._optimizer.state_dict()

        for it in range(self.attack_settings.iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(
                    it + 1))

            # Get metrics from victim model and substitute model
            self._logger.info("Getting substitute model's metrics for test "
                              "set")
            sub_test_acc = self._test_model(self._substitute_model,
                                            self._test_set)
            self._logger.info("Test set metrics")
            self._logger.info(
                    "Victim model Accuracy: {:.1f}%".format(vict_test_acc))
            self._logger.info(
                    "Substitute model Accuracy: {:.1f}%".format(sub_test_acc))

            # Reset substitute model and optimizer
            self._substitute_model.load_state_dict(sub_orig_state_dict)
            self._optimizer.load_state_dict(optim_orig_state_dict)

            # Step 3: The substitute model is trained with union of all the
            # labeled queried sets
            self._logger.info(
                    "Training substitute model with the query dataset")
            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, val_set, it + 1)

            if (it + 1) == (self.attack_settings.iterations + 1):
                self._finalize_attack()
                break

            self._get_aggreement_score()

            # Step 4: An Atlas subset selection strategy is used
            # to select set of k samples
            self._logger.info(
                    "Selecting {} samples from the remaining thief dataset"
                        .format(self.attack_settings.k))
            data_rest = Subset(self._thief_dataset, idxs_rest)
            idxs_query = self._atlas_strategy(data_rest, val_set)
            idxs_query = idxs_rest[idxs_query]
            idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
            query_set = Subset(self._thief_dataset, idxs_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            y_query = self._get_predictions(self._victim_model, query_set,
                                            self.attack_settings.output_type)
            query_sets.append(CustomLabelDataset(query_set, y_query))

        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()

        return
