import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchattacks import DeepFool
from tqdm import tqdm

from .base import Base
from ..utils.pytorch.datasets import CustomDataset, NoYDataset


class ActiveThief(Base):

    def __init__(self, victim_model, substitute_model, x_test, y_test,
                 num_classes, iterations=10, selection_strategy="entropy",
                 output_type="softmax", init_seed_size=20, budget=200,
                 training_epochs=1000, early_stop_tolerance=10,
                 evaluation_frequency=2, val_size=0.2, batch_size=64,
                 save_loc="./cache/activethief"):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        train_loss = F.mse_loss
        test_loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, x_test, y_test,
                         optimizer, train_loss, test_loss, training_epochs,
                         early_stop_tolerance, evaluation_frequency, val_size,
                         batch_size, num_classes, save_loc)

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

    def _entropy_strategy(self, k, idx_rest, y_rest):
        scores = {}
        for _, sample in enumerate(tqdm(zip(idx_rest, y_rest),
                                        desc="Calculating entropy scores")):
            sample_id, prob_dist = sample
            log_probs = prob_dist * np.log2(prob_dist)
            raw_entropy = 0 - np.sum(log_probs)

            normalized_entropy = raw_entropy / math.log2(prob_dist.size)

            scores[sample_id] = normalized_entropy

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return np.array(list(scores.keys())[:k])

    def _kcenter_strategy(self, k, idx_rest, y_rest, init_centers):
        y_rest = NoYDataset(y_rest)
        y_rest_loader = DataLoader(y_rest, batch_size=self._batch_size,
                                   num_workers=4, pin_memory=True)

        curr_centers = torch.from_numpy(init_centers)

        selected_points = []
        for _ in tqdm(range(k), desc="Selecting best points"):
            min_max_vals = []
            idx_min_max = []
            with torch.no_grad():
                for y_rest_batch in y_rest_loader:

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
            sample_id = batch_id * self._batch_size + idx_min_max[
                batch_id.item()]
            curr_centers = torch.from_numpy(
                    np.vstack([curr_centers.cpu(), y_rest[sample_id]]))
            selected_points.append(sample_id)

        return idx_rest[selected_points]

    def _deepfool_strategy(self, k, idx_rest, x_rest):
        self._substitute_model.eval()

        x_rest_data = NoYDataset(x_rest)
        x_rest_loader = DataLoader(x_rest_data, batch_size=self._batch_size,
                                   num_workers=4, pin_memory=True)

        deepfool = DeepFool(self._substitute_model, steps=3)

        scores = []
        for _, x in enumerate(
                tqdm(x_rest_loader, desc="Getting dfal scores")):

            x_adv = deepfool(x).cpu()

            # difference as L2-norm
            for adv, orig in zip(x_adv, x):
                scores.append(torch.dist(adv, orig).item())

        scores = dict(zip(idx_rest, scores))
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return np.array(list(scores.keys())[:k])

    def _select_samples(self, idx_rest, x_rest, y_rest, x_queries):
        if self._selection_strategy == "entropy":
            selected_samples = self._entropy_strategy(self._k, idx_rest,
                                                      y_rest)
        elif self._selection_strategy == "random":
            selected_samples = self._random_strategy(self._k, idx_rest)
        elif self._selection_strategy == "k-center":
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 x_queries)
            selected_samples = self._kcenter_strategy(self._k, idx_rest,
                                                      y_rest, init_centers)
        elif self._selection_strategy == "dfal":
            selected_samples = self._deepfool_strategy(self._k, idx_rest,
                                                       x_rest)
        elif self._selection_strategy == "dfal+k-center":
            dfal_top_idx = self._deepfool_strategy(self._budget, idx_rest,
                                                   x_rest)
            y_dfal_top = y_rest[dfal_top_idx]
            # Get initial centers
            init_centers = self._get_predictions(self._substitute_model,
                                                 x_queries)
            selected_samples = self._kcenter_strategy(self._k, dfal_top_idx,
                                                      y_dfal_top, init_centers)
        else:
            self._logger.warning(
                    "Selection strategy must be one of {entropy, random, "
                    "k-center, dfal, dfal+k-center}")
            raise ValueError

        return selected_samples

    def run(self, x, y):
        self._logger.info(
                "########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        self._logger.info(
                "ActiveThief's attack budget: {}".format(self._budget))

        idx_rest = np.arange(len(x))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idx_val = np.random.permutation(idx_rest)[:self._val_size]
        idx_rest = np.setdiff1d(idx_rest, idx_val)
        x_val = x[idx_val]
        y_val = self._get_predictions(self._victim_model, x_val,
                                      self._output_type)
        val_set = CustomDataset(x_val, y_val)

        val_label_counts = dict(list(enumerate([0] * self._num_classes)))
        for class_id in np.argmax(y_val, axis=1):
            val_label_counts[class_id] += 1

        self._logger.info("Validation dataset labels distribution: {}".format(
                val_label_counts))

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        x_queries = []
        y_queries = []

        idx_query = np.random.permutation(idx_rest)[:self._init_seed_size]
        idx_rest = np.setdiff1d(idx_rest, idx_query)
        x_queries.append(x[idx_query])

        y_query = self._get_predictions(self._victim_model, x_queries[0],
                                        self._output_type)
        y_queries.append(y_query)

        # Get victim model predicted labels for test set
        self._logger.info("Getting victim model's labels for test set")
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set.x)
        vict_test_labels = np.argmax(vict_test_labels, axis=1)
        self._logger.info(
                "Number of test samples: {}".format(len(vict_test_labels)))

        # Get victim model metrics on test set
        self._logger.info("Getting victim model's metrics for test set")
        vict_test_acc, vict_test_loss = self._test_model(self._victim_model,
                                                         self._test_loss,
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
                    self._substitute_model, self._test_loss, self._test_set)
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
            train_set = CustomDataset(np.vstack(x_queries),
                                      np.vstack(y_queries))
            self._train_model(self._substitute_model, self._optimizer,
                              self._train_loss, train_set, val_set, it)

            self._get_aggreement_score()

            # Step 4: Approximate labels are obtained for remaining samples
            # using the substitute

            x_rest = x[idx_rest]
            y_rest = None
            if self._selection_strategy not in {"random", "dfal"}:
                self._logger.info("Getting substitute's predictions for the "
                                  "rest of the thief dataset")
                y_rest = self._get_predictions(self._substitute_model, x_rest)

            # Step 5: An active learning subset selection strategy is used
            # to select set of k
            # samples
            self._logger.info("Selecting {} samples using the {} strategy from"
                              " the remaining thief dataset"
                              .format(self._k, self._selection_strategy))

            idx_query = self._select_samples(idx_rest, x_rest, y_rest,
                                             np.vstack(x_queries))
            idx_rest = np.setdiff1d(idx_rest, idx_query)
            x_queries.append(x[idx_query])

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            y_query = self._get_predictions(self._victim_model, x_queries[it],
                                            self._output_type)
            y_queries.append(y_query)
        return
