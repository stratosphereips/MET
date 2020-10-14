import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchattacks import DeepFool
from tqdm import tqdm

from .base import Base
from ..utils.pytorch.datasets import CustomLabelDataset


class ActiveThief(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 iterations=10, selection_strategy="entropy",
                 output_type="softmax", init_seed_size=20, budget=200,
                 training_epochs=1000, early_stop_tolerance=10,
                 evaluation_frequency=2, val_size=0.2, batch_size=64,
                 save_loc="./cache/activethief"):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        train_loss = F.mse_loss
        test_loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, train_loss,
                         test_loss, training_epochs, early_stop_tolerance,
                         evaluation_frequency, val_size, batch_size,
                         num_classes, save_loc)

        # BlackBox's specific attributes
        self._iterations = iterations
        self._selection_strategy = selection_strategy
        self._output_type = output_type
        self._init_seed_size = init_seed_size
        self._budget = budget
        self._val_size = int(self._budget * self._val_size)
        self._k = (self._budget - self._val_size - self._init_seed_size) // \
                  self._iterations

        if self._k <= 0:
            self._logger.error("ActiveThief's per iteration selection must "
                               "be bigger than 0!")
            raise ValueError()

        # Check configuration
        if self._selection_strategy not in ["random", "entropy", "k-center",
                                            "dfal", "dfal+k-center"]:
            self._logger.error(
                    "ActiveThief's selection strategy must be one of " +
                    "{random, entropy, k-center, dfal, dfal+kcenter}")
            raise ValueError()

    def _random_strategy(self, k, idx_rest):
        return np.random.permutation(idx_rest)[:k]

    def _entropy_strategy(self, k, idx_rest, data_rest):
        scores = {}
        for sample_id, sample in tqdm(zip(idx_rest, data_rest),
                                      total=len(data_rest),
                                      desc="Calculating entropy scores"):
            _, prob_dist = sample
            log_probs = prob_dist * torch.log2(prob_dist)
            raw_entropy = 0 - torch.sum(log_probs)

            normalized_entropy = raw_entropy / math.log2(self._num_classes)

            scores[sample_id] = normalized_entropy

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return np.array(list(scores.keys())[:k])

    def _kcenter_strategy(self, k, idx_rest, data_rest, init_centers):
        data_rest_loader = DataLoader(data_rest, batch_size=self._batch_size,
                                      num_workers=4, pin_memory=True)

        curr_centers = init_centers

        selected_points = []
        for _ in tqdm(range(k), desc="Selecting best points"):
            min_max_vals = []
            idx_min_max = []
            with torch.no_grad():
                for _, y_rest_batch in data_rest_loader:

                    if self._test_config.gpus:
                        y_rest_batch = y_rest_batch.cuda()
                        curr_centers = curr_centers.cuda()

                    dists = torch.cdist(y_rest_batch, curr_centers, p=2)
                    dists_min_vals, _ = torch.min(dists, dim=1)
                    dist_min_max_val, dist_min_max_id = torch.max(
                            dists_min_vals, dim=0)

                    min_max_vals.append(dist_min_max_val.cpu())
                    idx_min_max.append(dist_min_max_id.cpu())

            batch_id = np.argmax(min_max_vals)
            sample_id = batch_id * self._batch_size + \
                        idx_min_max[batch_id.item()]
            curr_centers = torch.cat([curr_centers.cpu(),
                                        data_rest.targets[sample_id][None, :]])
            selected_points.append(sample_id)

        return idx_rest[selected_points]

    def _deepfool_strategy(self, k, idx_rest, data_rest):
        self._substitute_model.eval()

        data_rest_loader = DataLoader(data_rest, batch_size=self._batch_size,
                                      num_workers=4, pin_memory=True)

        deepfool = DeepFool(self._substitute_model, steps=3)

        scores = []
        for x, _ in tqdm(data_rest_loader, desc="Getting dfal scores"):
            x_adv = deepfool(x).cpu()

            # difference as L2-norm
            for adv, orig in zip(x_adv, x):
                scores.append(torch.dist(adv, orig).item())

        scores = dict(zip(idx_rest, scores))
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return np.array(list(scores.keys())[:k])

    def _select_samples(self, idx_rest, data_rest, query_sets):
        if self._selection_strategy == "entropy":
            selected_samples = self._entropy_strategy(self._k, idx_rest,
                                                      data_rest)
        elif self._selection_strategy == "random":
            selected_samples = self._random_strategy(self._k, idx_rest)
        elif self._selection_strategy == "k-center":
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 query_sets)
            selected_samples = self._kcenter_strategy(self._k, idx_rest,
                                                      data_rest,
                                                      init_centers)
        elif self._selection_strategy == "dfal":
            selected_samples = self._deepfool_strategy(self._k, idx_rest,
                                                       data_rest)
        elif self._selection_strategy == "dfal+k-center":
            dfal_top_idx = self._deepfool_strategy(self._budget, idx_rest,
                                                   data_rest)
            y_dfal_top = data_rest.targets[dfal_top_idx]
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 query_sets)
            selected_samples = self._kcenter_strategy(self._k, dfal_top_idx,
                                                      y_dfal_top, init_centers)
        else:
            self._logger.warning("Selection strategy must be one of {entropy, "
                                 "random, k-center, dfal, dfal+k-center}")
            raise ValueError

        return selected_samples

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info(
                "########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        self._logger.info(
                "ActiveThief's attack budget: {}".format(self._budget))

        idx_rest = np.arange(len(self._sub_dataset))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idx_val = np.random.permutation(idx_rest)[:self._val_size]
        idx_rest = np.setdiff1d(idx_rest, idx_val)
        val_set = Subset(self._sub_dataset, idx_val)
        y_val = self._get_predictions(self._victim_model, val_set,
                                      self._output_type)
        val_set = CustomLabelDataset(val_set, y_val)

        val_label_counts = dict(list(enumerate([0] * self._num_classes)))
        for class_id in torch.argmax(y_val, dim=1):
            val_label_counts[class_id.item()] += 1

        self._logger.info("Validation dataset labels distribution: {}".format(
                val_label_counts))

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        query_sets = []

        idx_query = np.random.permutation(idx_rest)[:self._init_seed_size]
        idx_rest = np.setdiff1d(idx_rest, idx_query)
        query_set = Subset(self._sub_dataset, idx_query)
        y_query = self._get_predictions(self._victim_model, query_set,
                                        self._output_type)
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
        vict_test_acc, vict_test_loss = self._test_model(self._victim_model,
                                                         self._test_set)

        # Save substitute model state_dict for retraining from scratch
        sub_orig_state_dict = self._substitute_model.state_dict()
        optim_orig_state_dict = self._optimizer.state_dict()

        for it in range(1, self._iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it))

            # Get metrics from victim model and substitute model
            self._logger.info("Getting substitute model's metrics for test "
                              "set")
            sub_test_acc, sub_test_loss = self._test_model(
                    self._substitute_model, self._test_set)
            self._logger.info("Test set metrics")
            self._logger.info(
                    "Victim model Accuracy: {:.1f}% Loss: {:.3f}".format(
                            vict_test_acc, vict_test_loss))
            self._logger.info(
                    "Substitute model Accuracy: {:.1f}% Loss: {:.3f}".format(
                            sub_test_acc, sub_test_loss))

            # Reset substitute model and optimizer
            self._substitute_model.load_state_dict(sub_orig_state_dict)
            self._optimizer.load_state_dict(optim_orig_state_dict)

            # Step 3: The substitute model is trained with union of all the
            # labeled queried sets
            self._logger.info(
                    "Training substitute model with the query dataset")
            train_set = ConcatDataset(query_sets)
            self._train_model(self._substitute_model, self._optimizer,
                              train_set, val_set, it)

            self._get_aggreement_score()

            # Step 4: Approximate labels are obtained for remaining samples
            # using the substitute

            data_rest = Subset(self._sub_dataset, idx_rest)
            y_rest = None
            if self._selection_strategy not in {"random", "dfal"}:
                self._logger.info("Getting substitute's predictions for the "
                                  "rest of the thief dataset")
                y_rest = self._get_predictions(self._substitute_model,
                                               data_rest)

            if y_rest is not None:
                data_rest = CustomLabelDataset(data_rest, y_rest)

            # Step 5: An active learning subset selection strategy is used
            # to select set of k
            # samples
            self._logger.info("Selecting {} samples using the {} strategy from"
                              " the remaining thief dataset"
                              .format(self._k, self._selection_strategy))

            idx_query = self._select_samples(idx_rest, data_rest,
                                             ConcatDataset(query_sets))
            idx_rest = np.setdiff1d(idx_rest, idx_query)
            query_set = Subset(self._sub_dataset, idx_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            y_query = self._get_predictions(self._victim_model, query_set,
                                            self._output_type)
            query_sets.append(CustomLabelDataset(query_set, y_query))
        return
