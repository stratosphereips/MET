import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchattacks import DeepFool
from tqdm import tqdm

from .base import Base
from ..utils.pytorch.datasets import CustomLabelDataset, MefDataset
from ..utils.pytorch.functional import soft_cross_entropy
from ..utils.settings import AttackSettings


@dataclass
class ActiveThiefSettings(AttackSettings):
    iterations: int
    selection_strategy: str
    budget: int
    init_seed_size: int
    val_size: int
    k: int

    def __init__(self,
                 iterations: int,
                 selection_strategy: str,
                 budget: int):
        self.iterations = iterations
        self.selection_strategy = selection_strategy.lower()
        self.budget = budget

        # Check configuration
        if self.selection_strategy not in ["random", "entropy", "k-center",
                                           "dfal", "dfal+k-center"]:
            raise ValueError(
                    "ActiveThief's selection strategy must be one of {"
                    "random, entropy, k-center, dfal, dfal+k-center}")

        self.init_seed_size = int(self.budget * 0.1)
        self.val_size = int(self.budget * 0.2)
        self.k = (self.budget - self.val_size - self.init_seed_size) // \
                 self.iterations


class ActiveThief(Base):

    def __init__(self,
                 victim_model,
                 substitute_model,
                 num_classes,
                 iterations=10,
                 selection_strategy="entropy",
                 victim_output_type="prob_dist",
                 budget=20000,
                 optimizer: torch.optim.Optimizer = None,
                 loss=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(substitute_model.parameters())
        if loss is None:
            loss = soft_cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss,
                         num_classes, victim_output_type)
        self.attack_settings = ActiveThiefSettings(iterations,
                                                   selection_strategy, budget)

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="ActiveThief attack")
        parser.add_argument("--selection_strategy", default="entropy",
                            type=str,
                            help="Activethief selection strategy can "
                                 "be one of {random, entropy, k-center, "
                                 "dfal, dfal+k-center} (Default: "
                                 "entropy)")
        parser.add_argument("--iterations", default=10, type=int,
                            help="Number of iterations of the attacks ("
                                 "Default: "
                                 "10)")
        parser.add_argument("--victim_output_type", default="prob_dist",
                            type=str,
                            help="Type of output from victim model {"
                                 "prob_dist, raw, one_hot} (Default: "
                                 "prob_dist)")
        parser.add_argument("--budget", default=20000, type=int,
                            help="Size of the budget (Default: 20000)")
        parser.add_argument("--training_epochs", default=1000,
                            type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 1000)")
        parser.add_argument("--patience", default=100, type=int,
                            help="Number of epochs without improvement for "
                                 "early "
                                 "stop (Default: 100)")
        parser.add_argument("--evaluation_frequency", default=1, type=int,
                            help="Epochs interval of validation (Default: 1)")

        cls._add_base_args(parser)

        return parser

    def _random_strategy(self,
                         k,
                         data_rest):
        return np.random.permutation(len(data_rest))[:k]

    def _entropy_strategy(self,
                          k,
                          data_rest):
        scores = []
        data_rest = MefDataset(self.base_settings, data_rest)
        loader = data_rest.generic_dataloader()
        for _, prob_dists in tqdm(loader, desc="Calculating entropy scores"):
            log_probs = prob_dists * torch.log2(prob_dists)
            raw_entropy = 0 - torch.sum(log_probs, dim=1)

            normalized_entropy = raw_entropy / math.log2(
                    self._num_classes)

            scores.append(normalized_entropy)

        return torch.cat(scores).topk(k).indices.numpy()

    def _kcenter_strategy(self,
                          k,
                          data_rest,
                          init_centers):
        data_rest = MefDataset(self.base_settings, data_rest)
        loader = data_rest.generic_dataloader()

        if self.base_settings.gpus:
            init_centers = init_centers.cuda()

        min_dists = []
        with torch.no_grad():
            for _, y_rest_batch in tqdm(loader, desc="Calculating distance "
                                                     "from initial centers"):
                if self.base_settings.gpus:
                    y_rest_batch = y_rest_batch.cuda()

                # To save memory we are only keeping the minimal distance
                # for each y from initial centers
                batch_dists = torch.cdist(y_rest_batch, init_centers, p=2)
                batch_dists_min_vals, _ = torch.min(batch_dists, dim=-1)
                min_dists.append(batch_dists_min_vals)

        min_dists = torch.cat(min_dists)
        selected_points = []
        for _ in tqdm(range(k // 5), desc="Selecting best points"):
            min_dists_max_ids = torch.argsort(min_dists, dim=-1,
                                              descending=True)[:5]

            selected_points.append(min_dists_max_ids)
            new_centers = data_rest.train_set.targets[min_dists_max_ids]

            if self.base_settings.gpus:
                new_centers = new_centers.cuda()

            new_centers_dists_min_vals = []
            with torch.no_grad():
                for _, y_rest_batch in loader:

                    if self.base_settings.gpus:
                        y_rest_batch = y_rest_batch.cuda()

                    batch_dists = torch.cdist(y_rest_batch, new_centers, p=2)
                    batch_dists_min_vals, _ = torch.min(batch_dists, dim=-1)

                    new_centers_dists_min_vals.append(batch_dists_min_vals)

                min_dists = torch.stack(
                        [min_dists, torch.cat(new_centers_dists_min_vals)],
                        dim=1)
                # For each y we update minimal distance
                min_dists, _ = torch.min(min_dists, dim=-1)

        return torch.stack(selected_points).detach().cpu().numpy()

    def _deepfool_strategy(self,
                           k,
                           data_rest):
        self._substitute_model.eval()
        data_rest = MefDataset(self.base_settings, data_rest)
        loader = data_rest.generic_dataloader()

        deepfool = DeepFool(self._substitute_model, steps=30)

        scores = []
        for x, _ in tqdm(loader, desc="Getting dfal scores"):
            if self.base_settings.gpus:
                x = x.cuda()

            x_adv = deepfool(x)

            # difference as L2-norm
            scores.append(torch.dist(x_adv, x))

        return torch.stack(scores).topk(k).indices.detach().cpu().numpy()

    def _select_samples(self,
                        data_rest,
                        query_sets):
        selection_strategy = self.attack_settings.selection_strategy
        budget = self.attack_settings.budget
        k = self.attack_settings.k

        self._logger.info("Selecting {} samples using the {} strategy from"
                          " the remaining thief dataset"
                          .format(k, selection_strategy))

        if selection_strategy == "entropy":
            selected_points = self._entropy_strategy(k, data_rest)
        elif selection_strategy == "random":
            selected_points = self._random_strategy(k, data_rest)
        elif selection_strategy == "k-center":
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 query_sets)
            selected_points = self._kcenter_strategy(k, data_rest,
                                                     init_centers)
        elif selection_strategy == "dfal":
            selected_points = self._deepfool_strategy(k, data_rest)
        elif selection_strategy == "dfal+k-center":
            idxs_dfal_best = self._deepfool_strategy(budget, data_rest)
            y_dfal_best = data_rest.targets[idxs_dfal_best]
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 query_sets)
            idxs_kcenter_best = self._kcenter_strategy(k, y_dfal_best,
                                                       init_centers)
            selected_points = idxs_dfal_best[idxs_kcenter_best]
        else:
            self._logger.warning("Selection strategy must be one of {entropy, "
                                 "random, k-center, dfal, dfal+k-center}")
            raise ValueError

        return selected_points

    def _run(self):
        self._logger.info(
                "########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        self._logger.info("ActiveThief's attack budget: {}"
                          .format(self.attack_settings.budget))

        idxs_rest = np.arange(len(self._thief_dataset))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idxs_val = np.random.permutation(idxs_rest)[
                   : self.attack_settings.val_size]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_val)
        val_set = Subset(self._thief_dataset, idxs_val)
        y_val = self._get_predictions(self._victim_model, val_set)
        val_set = CustomLabelDataset(val_set, y_val)

        val_label_counts = dict(list(enumerate([0] * self._num_classes)))
        if len(y_val.size()) == 1:
            for class_id in torch.round(y_val):
                val_label_counts[class_id.item()] += 1
        else:
            for class_id in torch.argmax(y_val, dim=-1):
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
        y_query = self._get_predictions(self._victim_model, query_set)
        query_sets.append(CustomLabelDataset(query_set, y_query))

        # Get victim model metrics on test set
        self._logger.info("Getting victim model's metrics for test set")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)

        # Save substitute model state_dict for retraining from scratch
        sub_orig_state_dict = self._substitute_model.state_dict()
        optim_orig_state_dict = self._substitute_model.optimizer.state_dict()

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
            self._substitute_model.optimizer.load_state_dict(
                    optim_orig_state_dict)

            # Step 3: The substitute model is trained with union of all the
            # labeled queried sets
            self._logger.info(
                    "Training substitute model with the query dataset")
            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, val_set, it + 1)

            if (it + 1) == (self.attack_settings.iterations + 1):
                break

            self._get_aggreement_score()

            # Step 4: Approximate labels are obtained for remaining samples
            # using the substitute
            data_rest = Subset(self._thief_dataset, idxs_rest)
            # Random and dfal strategies dont require predictions for the rest
            # of thief dataset
            if self.attack_settings.selection_strategy not in {
                "random", "dfal"}:
                self._logger.info("Getting substitute's predictions for the "
                                  "rest of the thief dataset")
                y_rest = self._get_predictions(self._substitute_model,
                                               data_rest)
                data_rest = CustomLabelDataset(data_rest, y_rest)

            # Step 5: An active learning subset selection strategy is used
            # to select set of k samples
            idxs_query = self._select_samples(data_rest,
                                              ConcatDataset(query_sets))
            idxs_query = idxs_rest[np.unique(idxs_query)]
            idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
            query_set = Subset(self._thief_dataset, idxs_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            y_query = self._get_predictions(self._victim_model, query_set)
            query_sets.append(CustomLabelDataset(query_set, y_query))

        return
