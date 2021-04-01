import pickle
from argparse import ArgumentParser
from collections import defaultdict as dd
from dataclasses import dataclass
from typing import Optional, Tuple

import foolbox as fb
import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from .base import Base
from ..utils.pytorch.datasets import CustomLabelDataset, NoYDataset
from ..utils.pytorch.functional import get_class_labels, get_prob_vector
from ..utils.pytorch.lighting.module import TrainableModel, VictimModel
from ..utils.settings import AttackSettings


@dataclass
class ActiveThiefSettings(AttackSettings):
    iterations: int
    selection_strategy: str
    budget: int
    init_seed_size: int
    val_size: int
    k: int
    save_samples: bool

    def __init__(
        self,
        iterations: int,
        selection_strategy: str,
        budget: int,
        init_seed_size: float,
        val_size: float,
        bounds: Tuple[float, float],
        save_samples: bool,
    ):
        self.iterations = iterations
        self.selection_strategy = selection_strategy.lower()
        self.budget = budget
        self.bounds = bounds
        self.save_samples = save_samples

        # Check configuration
        if self.selection_strategy not in [
            "random",
            "entropy",
            "k-center",
            "dfal",
            "dfal+k-center",
        ]:
            raise ValueError(
                "ActiveThief's selection strategy must be one of {"
                "random, entropy, k-center, dfal, dfal+k-center}"
            )

        self.init_seed_size = int(self.budget * init_seed_size)
        self.val_size = int(self.budget * val_size)
        self.k = (self.budget - self.val_size - self.init_seed_size) // self.iterations


class ActiveThief(Base):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        iterations: int = 10,
        selection_strategy: str = "entropy",
        budget: int = 20000,
        init_seed_size: float = 0.1,
        val_size: float = 0.2,
        bounds: Tuple[float, float] = (0, 1),
        save_samples: bool = False,
    ):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = ActiveThiefSettings(
            iterations,
            selection_strategy,
            budget,
            init_seed_size,
            val_size,
            bounds,
            save_samples,
        )
        self._val_dataset = None
        self._selected_samples = dd(list)

    @classmethod
    def _get_attack_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(description="ActiveThief attack")
        parser.add_argument(
            "--selection_strategy",
            default="entropy",
            type=str,
            help="Activethief selection strategy can "
            "be one of {random, entropy, k-center, "
            "dfal, dfal+k-center} (Default: "
            "entropy)",
        )
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
        parser.add_argument(
            "--init_seed_size",
            default=0.1,
            type=float,
            help="Fraction of budget that should be used for "
            "initial random query (Default: 0.1)",
        )
        parser.add_argument(
            "--val_size",
            default=0.2,
            type=float,
            help="Fraction of budget that should be used for "
            "validation set (Default: 0.2)",
        )
        parser.add_argument(
            "--idxs",
            action="store_true",
            help="Whether to save idxs of samples selected "
            "during the attacks. (Default: False)",
        )

        return parser

    def _random_strategy(self, k: int, data_rest: CustomLabelDataset) -> np.ndarray:
        return np.random.permutation(len(data_rest))[:k]

    def _entropy_strategy(self, k: int, data_rest: CustomLabelDataset) -> np.ndarray:
        scores = Categorical(data_rest.targets).entropy()
        return scores.argsort(descending=True)[:k].numpy()

    def _kcenter_strategy(
        self, k: int, preds_sub_rest: torch.Tensor, init_centers: torch.Tensor
    ) -> np.ndarray:
        loader = DataLoader(
            dataset=NoYDataset(preds_sub_rest),
            pin_memory=self.base_settings.gpu,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        if self.base_settings.gpu:
            init_centers = init_centers.cuda()

        min_dists = []
        with torch.no_grad():
            for preds_rest_batch, _ in tqdm(loader):
                if self.base_settings.gpu:
                    preds_rest_batch = preds_rest_batch.cuda()

                # To save memory we are only keeping the minimal distance
                # for each y from initial centers
                batch_dists = torch.cdist(preds_rest_batch, init_centers, p=2)
                batch_dists_min_vals, _ = torch.min(batch_dists, dim=-1)
                min_dists.append(batch_dists_min_vals)

        min_dists = torch.cat(min_dists)
        selected_points = []
        # In the paper they are selecting one center in each iteration,
        # this is however extremely slow even with optimization. We thus
        # select 5 samples each iteration
        for _ in tqdm(range(k // 5), desc="Selecting best points"):
            min_dists_max_ids = torch.argsort(min_dists, dim=-1, descending=True)[:5]

            selected_points.append(min_dists_max_ids)
            new_centers = preds_sub_rest[min_dists_max_ids]

            if self.base_settings.gpu:
                new_centers = new_centers.cuda()

            new_centers_dists_min_vals = []
            with torch.no_grad():
                for preds_rest_batch, _ in loader:

                    if self.base_settings.gpu:
                        preds_rest_batch = preds_rest_batch.cuda()

                    batch_dists = torch.cdist(preds_rest_batch, new_centers, p=2)
                    batch_dists_min_vals, _ = torch.min(batch_dists, dim=-1)

                    new_centers_dists_min_vals.append(batch_dists_min_vals)

                min_dists = torch.stack(
                    [min_dists, torch.cat(new_centers_dists_min_vals)], dim=1
                )
                # For each y we update minimal distance
                min_dists, _ = torch.min(min_dists, dim=-1)

        return torch.stack(selected_points).detach().cpu().numpy()

    def _deepfool_strategy(self, k: int, data_rest: CustomLabelDataset) -> np.ndarray:
        self._substitute_model.eval()
        loader = DataLoader(
            dataset=data_rest,
            pin_memory=self.base_settings.gpu,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        fmodel = fb.PyTorchModel(
            self._substitute_model.model, bounds=self.attack_settings.bounds
        )
        deepfool = fb.attacks.L2DeepFoolAttack(steps=30, candidates=3, overshoot=0.01)

        scores = []
        for x, y in tqdm(loader, desc="Getting dfal scores"):
            if self.base_settings.gpu:
                x = x.cuda()
                y = y.cuda()

            labels = get_class_labels(y)
            criterion = fb.criteria.Misclassification(labels)
            x_adv, _, _ = deepfool(fmodel, x, criterion, epsilons=1)

            # difference as L2-norm
            for el1, el2 in zip(x_adv, x):
                scores.append(torch.dist(el1, el2).detach().cpu())

        return torch.stack(scores).argsort()[:k].numpy()

    def _select_samples(
        self, data_rest: CustomLabelDataset, query_sets: ConcatDataset
    ) -> np.ndarray:
        selection_strategy = self.attack_settings.selection_strategy
        budget = self.attack_settings.budget
        k = self.attack_settings.k

        self._logger.info(
            "Selecting {} samples using the {} strategy from"
            " the remaining thief dataset".format(k, selection_strategy)
        )

        if selection_strategy == "entropy":
            selected_points = self._entropy_strategy(k, data_rest)
        elif selection_strategy == "random":
            selected_points = self._random_strategy(k, data_rest)
        elif selection_strategy == "k-center":
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model, query_sets)
            # Substitute model returns logits
            init_centers = get_prob_vector(init_centers)
            selected_points = self._kcenter_strategy(k, data_rest.targets, init_centers)
        elif selection_strategy == "dfal":
            selected_points = self._deepfool_strategy(k, data_rest)
        elif selection_strategy == "dfal+k-center":
            idxs_dfal_best = self._deepfool_strategy(budget, data_rest)
            preds_sub_dfal_best = data_rest.targets[idxs_dfal_best]
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model, query_sets)
            # Substitute model returns logits
            init_centers = get_prob_vector(init_centers)
            idxs_kcenter_best = self._kcenter_strategy(
                k, preds_sub_dfal_best, init_centers
            )
            selected_points = idxs_dfal_best[idxs_kcenter_best]
        else:
            self._logger.warning(
                "Selection strategy must be one of {entropy, "
                "random, k-center, dfal, dfal+k-center}"
            )
            raise ValueError

        return selected_points

    def _check_args(
        self, sub_data: Dataset, test_set: Dataset, val_data: Dataset
    ) -> None:
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's " "dataset.")
            raise TypeError()
        self._thief_dataset = sub_data

        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()
        self._test_set = test_set

        if val_data is not None:
            if not isinstance(val_data, Dataset):
                self._logger.error("Validation dataset must be Pytorch's " "dataset.")
                raise TypeError()
        self._val_dataset = val_data

        return

    def _run(
        self, sub_data: Dataset, test_set: Dataset, val_data: Optional[Dataset] = None
    ) -> None:
        self._check_args(sub_data, test_set, val_data)
        self._logger.info("########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        self._logger.info(
            "ActiveThief's attack budget: {}".format(self.attack_settings.budget)
        )

        idxs_rest = np.arange(len(self._thief_dataset))

        # Prepare validation set
        val_set = None
        if self.trainer_settings.evaluation_frequency is not None:
            self._logger.info("Preparing validation dataset")
            if self._val_dataset is None:
                idxs_val = np.random.permutation(idxs_rest)[
                    : self.attack_settings.val_size
                ]
                idxs_rest = np.setdiff1d(idxs_rest, idxs_val)
                val_set = Subset(self._thief_dataset, idxs_val)
                y_val = self._get_predictions(self._victim_model, val_set)
            else:
                idxs_val = np.arange(len(self._val_dataset))
                idxs_val = np.random.permutation(idxs_val)[
                    : self.attack_settings.val_size
                ]
                val_set = Subset(self._val_dataset, idxs_val)
                y_val = self._get_predictions(self._victim_model, val_set)

            val_set = CustomLabelDataset(val_set, y_val)
            if self.attack_settings.save_samples:
                self._selected_samples["val_data"].extend(idxs_val)

            val_label_counts = dict(
                list(enumerate([0] * self._victim_model.num_classes))
            )
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
        query_set = Subset(self._thief_dataset, idxs_query)
        y_query = self._get_predictions(self._victim_model, query_set)
        query_sets.append(CustomLabelDataset(query_set, y_query))
        if self.attack_settings.save_samples:
            self._selected_samples["idxs"].extend(idxs_query)
            self._selected_samples["labels"].append(y_query)

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
            self._logger.info("Getting substitute model metrics for test set")
            sub_test_acc = self._test_model(self._substitute_model, self._test_set)
            self._logger.info("Test set metrics")
            self._logger.info("Victim model Accuracy: {:.1f}%".format(vict_test_acc))
            self._logger.info("Substitute model Accuracy: {:.1f}%".format(sub_test_acc))
            self._get_aggreement_score()

            # Step 4: Approximate labels are obtained for remaining samples
            # using the substitute
            data_rest = Subset(self._thief_dataset, idxs_rest)
            # Random strategy doesn't require predictions for the rest
            # of thief dataset
            if self.attack_settings.selection_strategy != "random":
                self._logger.info(
                    "Getting substitute's predictions for the "
                    "rest of the thief dataset"
                )
                y_rest = self._get_predictions(self._substitute_model, data_rest)
                # Substitute model returns logits
                y_rest = get_prob_vector(y_rest)
                data_rest = CustomLabelDataset(data_rest, y_rest)

            # Step 5: An active learning subset selection strategy is used
            # to select set of k samples
            idxs_query = self._select_samples(data_rest, ConcatDataset(query_sets))
            idxs_query = idxs_rest[np.unique(idxs_query)]
            idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
            query_set = Subset(self._thief_dataset, idxs_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info(
                "Getting predictions for the current query set " "from the victim model"
            )
            y_query = self._get_predictions(self._victim_model, query_set)
            query_sets.append(CustomLabelDataset(query_set, y_query))

            if self.attack_settings.save_samples:
                self._selected_samples["idxs"].extend(idxs_query)
                self._selected_samples["labels"].append(y_query)

        if self.attack_settings.save_samples:
            filepath = self.base_settings.save_loc.joinpath("selected_samples.pl")
            self._selected_samples["labels"] = torch.cat(
                self._selected_samples["labels"]
            )
            with open(filepath, "wb") as f:
                pickle.dump(self._selected_samples, f)

        return
