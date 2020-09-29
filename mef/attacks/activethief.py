import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from .base import Base
from ..utils.config import Configuration
from ..utils.pytorch.datasets import CustomLabelDataset, split_data


class DeepFool:
    """Batch deepfool
    https://github.com/tobylyf/adv-attack/blob/master/mydeepfool.py"""

    def __init__(self, model, nb_candidate, overshoot=0.02, max_iter=50,
                 clip_min=0.0, clip_max=1.0, force_max_iter=False):
        self._model = model
        self._nb_candidate = nb_candidate
        self._overshoot = overshoot
        self._max_iter = max_iter
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._force_max_iter = force_max_iter
        self._device = next(model.parameters()).device

    @staticmethod
    def _jacobian(predictions, x, nb_classes):
        list_derivatives = []

        for class_ind in range(nb_classes):
            outputs = predictions[:, class_ind]
            derivatives, = torch.autograd.grad(outputs, x,
                                               grad_outputs=torch.ones_like(
                                                       outputs),
                                               retain_graph=True)
            list_derivatives.append(derivatives)

        return list_derivatives

    def attack(self, model, x):
        with torch.no_grad():
            logits = model(x)
        nb_classes = logits.size(-1)
        assert self._nb_candidate <= nb_classes, "nb_candidate should not " \
                                                 "be greater than nb_classes"

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()

        iteration = 0
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(self._device)
        r_tot = torch.zeros(x.size()).to(self._device)
        original = current

        while (current == original).any and (
                self._force_max_iter or iteration < self._max_iter):
            predictions_val = logits.topk(self._nb_candidate)[0]
            gradients = torch.stack(self._jacobian(predictions_val, adv_x,
                                                   self._nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self._nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        # Calculate distance to the hyperplane
                        # Added 1e-4 for numerical stability
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    # Calculate minimal vector (perturbation) that projects
                    # sample onto the
                    # closest hyperplane
                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self._clip_min,
                                self._clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        adv_x = (1 + self._overshoot) * r_tot + x

        return adv_x


@dataclass
class ActiveThiefConfig:
    selection_strategy: str = "entropy"
    budget: int = 20000
    iterations: int = 10
    initial_seed_size: int = 2000
    output_type: str = "one_hot"


class ActiveThief(Base):

    def __init__(self, victim_model, substitute_model, test_set, thief_dataset,
                 num_classes, save_loc="./cache/activethief"):

        # Get ActiveThief's configuration
        self._config = Configuration.get_configuration(ActiveThiefConfig,
                                                       "attacks/activethief")
        self._selection_strategy = self._config.selection_strategy.lower()

        super().__init__(save_loc)

        # Datasets
        self._test_set = test_set
        self._thief_dataset = thief_dataset

        # Dataset information
        self._num_classes = num_classes
        self._budget = self._config.budget
        self._val_size = int(self._budget * self._test_config.val_set_size)
        self._k = (self._budget - self._val_size -
                   self._config.initial_seed_size) // self._config.iterations

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        if self._test_config.gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        # Optimizer, loss_functions
        self._optimizer = torch.optim.Adam(self._substitute_model.parameters())
        self._train_loss = F.mse_loss
        self._test_loss = F.cross_entropy

        # Check configuration
        if self._selection_strategy not in ["random", "entropy", "k-center",
                                            "dfal", "dfal+k-center"]:
            self._logger.error(
                    "ActiveThief's selection strategy must be one of " +
                    "{random, entropy, k-center, dfal, dfal+kcenter}")
            raise ValueError()

    def _random_strategy(self, k, idx):
        idx_copy = np.copy(idx)
        np.random.shuffle(idx_copy)

        return list(idx_copy[:k])

    def _entropy_strategy(self, k, idx, remaining_samples_predictions):
        scores = {}
        for _, sample in enumerate(
                tqdm(zip(idx, remaining_samples_predictions),
                     desc="Calculating entropy scores")):
            sample_id, prob_dist = sample
            log_probs = prob_dist * torch.log2(prob_dist)
            raw_entropy = 0 - torch.sum(log_probs)

            normalized_entropy = raw_entropy / math.log2(prob_dist.numel())

            scores[sample_id] = normalized_entropy.item()

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return list(scores.keys())[:k]

    def _kcenter_strategy(self, k, idx, remaining_samples_predictions,
                          query_sets_predictions):
        loader = DataLoader(remaining_samples_predictions,
                            batch_size=self._test_config.batch_size,
                            num_workers=4, pin_memory=True)

        centers_pred = query_sets_predictions
        selected_points = []
        for _ in tqdm(range(k), desc="Selecting best points"):
            min_max_values = []
            min_max_idx = []
            with torch.no_grad():
                for batch in loader:
                    samples_pred = batch
                    centers_pred = centers_pred

                    if self._test_config.gpus:
                        samples_pred = samples_pred.cuda()
                        centers_pred = centers_pred.cuda()

                    distances = torch.cdist(samples_pred, centers_pred, p=2)
                    distances_min_values, _ = torch.min(distances, dim=1)
                    distance_min_max_value, distance_min_max_id = torch.max(
                            distances_min_values, dim=0)

                    min_max_values.append(distance_min_max_value.cpu())
                    min_max_idx.append(distance_min_max_id.cpu())

            batch_id = np.argmax(min_max_values)
            sample_id = batch_id * self._test_config.batch_size + min_max_idx[
                batch_id.item()]
            centers_pred = torch.from_numpy(
                    np.vstack([centers_pred.cpu(),
                               remaining_samples_predictions[sample_id]]))

            selected_points.append(sample_id)

        return [idx[point] for point in selected_points]

    def _deepfool_strategy(self, k, idx, remaining_samples):
        self._substitute_model.eval()
        loader = DataLoader(remaining_samples, num_workers=4, pin_memory=True,
                            batch_size=self._test_config.batch_size)

        df = DeepFool(self._substitute_model, self._num_classes)

        scores = []
        for _, batch in enumerate(
                tqdm(loader, desc="Getting samples dfal scores")):
            x = batch[0]

            adversary_x = df.attack(self._substitute_model, x)

            batch_scores = []
            for adversary, original in zip(adversary_x, x):
                scores.append(torch.dist(adversary, original).item())

            scores.extend(batch_scores)

        scores = dict(zip(idx, scores))
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return list(scores.keys())[:k]

    def _select_samples(self, idx, remaining_samples, remaining_samples_preds,
                        query_sets):
        if self._selection_strategy == "entropy":
            selected_samples = self._entropy_strategy(self._k, idx,
                                                      remaining_samples_preds)
        elif self._selection_strategy == "random":
            selected_samples = self._random_strategy(self._k, idx)
        elif self._selection_strategy == "k-center":
            # Get initial centers
            query_sets_predictions = self._get_predictions(
                    self._substitute_model, query_sets)
            selected_samples = self._kcenter_strategy(self._k, idx,
                                                      remaining_samples_preds,
                                                      query_sets_predictions)
        elif self._selection_strategy == "dfal":
            selected_samples = self._deepfool_strategy(self._k, idx,
                                                       remaining_samples)
        elif self._selection_strategy == "dfal+k-center":
            dfal_selected_samples_idx = self._deepfool_strategy(self._budget,
                                                                idx,
                                                                remaining_samples)
            sorter = np.argsort(idx)
            ssdl_predictions_idx = sorter[
                np.searchsorted(idx, dfal_selected_samples_idx, sorter=sorter)]
            ssdl_predictions = remaining_samples_preds[ssdl_predictions_idx]
            # Get initial centers
            query_sets_predictions = self._get_predictions(
                    self._substitute_model, query_sets)
            selected_samples = self._kcenter_strategy(self._k,
                                                      dfal_selected_samples_idx,
                                                      ssdl_predictions,
                                                      query_sets_predictions)
        else:
            self._logger.warning(
                    "Selection strategy must be one of {entropy, random, "
                    "k-center, "
                    "dfal, dfal+k-center}")
            raise ValueError

        return selected_samples

    def run(self):
        self._logger.info(
                "########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        self._logger.info(
                "ActiveThief's attack budget: {}".format(self._budget))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        thief_dataset_rest, val_set = split_data(self._thief_dataset,
                                                 self._val_size)

        validation_predictions = self._get_predictions(self._victim_model,
                                                       val_set,
                                                       self._config.output_type)
        val_set_with_vict_pred = CustomLabelDataset(val_set,
                                                    validation_predictions)

        val_label_counts = dict(list(enumerate([0] * self._num_classes)))
        for class_id in torch.argmax(validation_predictions, dim=1):
            val_label_counts[class_id.item()] += 1

        self._logger.info("Validation dataset labels distribution: {}".format(
                val_label_counts))

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        available_samples = set(range(len(thief_dataset_rest)))
        query_sets = []
        query_sets_preds = []

        idx = np.random.choice(np.arange(len(available_samples)),
                               size=self._config.initial_seed_size,
                               replace=False)
        available_samples -= set(idx)

        query_sets.append(Subset(thief_dataset_rest, idx))
        query_set_preds = self._get_predictions(self._victim_model,
                                                query_sets[0],
                                                self._config.output_type)
        query_sets_preds.append(query_set_preds)

        # Get victim model predicted labels for test set
        self._logger.info("Getting victim model's labels for test set")
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=1).numpy()
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

        for iteration in range(1, self._config.iterations + 1):
            self._logger.info(
                    "---------- Iteration: {} ----------".format(iteration))

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
            query_set = CustomLabelDataset(ConcatDataset(query_sets),
                                           torch.cat(query_sets_preds))
            self._train_model(self._substitute_model, self._optimizer,
                              self._train_loss, query_set,
                              val_set_with_vict_pred, iteration)

            # Agreement score
            self._logger.info("Getting attack metric")
            self._get_attack_metric(self._substitute_model, self._test_set,
                                    vict_test_labels)

            # Step 4: Approximate labels are obtained for remaining samples
            # using the substitute
            self._logger.info("Getting substitute's predictions for the rest "
                              "of the thief dataset")
            idx = sorted(list(available_samples))
            remaining_samples = Subset(thief_dataset_rest, idx)
            remaining_samples_preds = self._get_predictions(
                    self._substitute_model, remaining_samples)

            # Step 5: An active learning subset selection strategy is used
            # to select set of k
            # samples
            self._logger.info(
                    "Selecting {} samples using the {} strategy from the "
                    "remaining thief dataset"
                        .format(self._k, self._selection_strategy))
            selected_samples = self._select_samples(idx, remaining_samples,
                                                    remaining_samples_preds,
                                                    ConcatDataset(query_sets))
            available_samples -= set(selected_samples)
            query_sets.append(Subset(thief_dataset_rest, selected_samples))

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            query_set_preds = self._get_predictions(self._victim_model,
                                                    query_sets[iteration],
                                                    self._config.output_type)
            query_sets_preds.append(query_set_preds)
