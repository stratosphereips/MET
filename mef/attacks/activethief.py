import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator, Events, create_supervised_trainer, Engine
from ignite.handlers import Checkpoint, EarlyStopping, global_step_from_engine, DiskSaver
from ignite.metrics import Fbeta, RunningAverage
from ignite.utils import to_onehot
from torch.utils.data import Sampler, DataLoader, ConcatDataset

from mef.attacks.base import Base
from mef.utils.config import Configuration
from mef.utils.pytorch.datasets import CustomLabelDataset
from mef.utils.pytorch.ignite.metrics import MacroAccuracy


class DeepFool(object):
    """Batch deepfool https://github.com/tobylyf/adv-attack/blob/master/mydeepfool.py"""

    def __init__(self, nb_candidate, overshoot=0.02, max_iter=50, clip_min=0.0, clip_max=1.0,
                 device="cpu"):
        self._nb_candidate = nb_candidate
        self._overshoot = overshoot
        self._max_iter = max_iter
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._device = device

    @staticmethod
    def _jacobian(predictions, x, nb_classes):
        list_derivatives = []

        for class_ind in range(nb_classes):
            outputs = predictions[:, class_ind]
            derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                                               retain_graph=True)
            list_derivatives.append(derivatives)

        return list_derivatives

    def attack(self, model, x):
        with torch.no_grad():
            logits = model(x)
        nb_classes = logits.size(-1)
        assert self._nb_candidate <= nb_classes, 'nb_candidate should not be greater than ' \
                                                 'nb_classes'

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

        while ((current == original).any and iteration < self._max_iter):
            predictions_val = logits.topk(self._nb_candidate)[0]
            gradients = torch.stack(self._jacobian(predictions_val, adv_x, self._nb_candidate),
                                    dim=1)
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

                    # Calculate minimal vector (perturbation) that projects sample onto the
                    # closest hyperplane
                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self._clip_min, self._clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        adv_x = (1 + self._overshoot) * r_tot + x

        return adv_x


class Dfal(nn.Module):
    def __init__(self, model, num_candidates, iterations=20):
        super().__init__()
        self._model = model
        self._device = next(model.parameters()).device
        self._deepfool = DeepFool(num_candidates, max_iter=iterations, device=self._device)

    def forward(self, x):
        x = x.to(self._device)

        adversary_x = self._deepfool.attack(self._model, x)

        scores = []
        for adversary, original in zip(adversary_x, x):
            scores.append(torch.dist(adversary, original).item())

        return scores


class KCenter(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self._device = device

    def forward(self, x, y):
        x = x.to(self._device)
        y = y.to(self._device)

        distances = torch.cdist(x, y, p=2)

        distances_min_values, _ = torch.min(distances, dim=1)
        distance_min_max_value, distance_min_max_id = torch.max(distances_min_values, dim=0)

        return distance_min_max_value.cpu(), distance_min_max_id.cpu()


@dataclass
class ActiveThiefConfig:
    selection_strategy: str = "entropy"
    k: int = 1500
    iterations: int = 10
    training_epochs: int = 1000
    initial_seed_size: int = 2000
    one_hot_output: bool = False


class ActiveThief(Base):

    def __init__(self, secret_model, substitute_model, test_dataset, thief_dataset,
                 validation_dataset, save_loc="./cache/activethief"):
        super().__init__()

        # Get ActiveThief's configuration
        self._config = Configuration.get_configuration(ActiveThiefConfig, "attacks/activethief")
        self._device = "cuda" if self._test_config.gpu is not None else "cpu"
        self._save_loc = save_loc

        # Datasets
        self._test_dataset = test_dataset
        self._thief_dataset = thief_dataset
        self._validation_dataset_original = validation_dataset
        self._validation_dataset_predicted = None

        # Models
        self._secret_model = secret_model
        self._substitute_model = substitute_model

        # Dataset information
        self._num_classes = self._secret_model.num_classes

    def _get_predictions(self, model, data, one_hot=False):
        loader = DataLoader(data, pin_memory=True, num_workers=4, batch_size=256)

        evaluator = create_supervised_evaluator(model, device=self._device)
        evaluator.logger = self._logger
        ProgressBar().attach(evaluator)

        evaluator.state.y_preds = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def append_stolen_labels(evaluator):
            evaluator.state.y_preds.append(evaluator.state.output[0])

        @evaluator.on(Events.COMPLETED)
        def append_stolen_labels(evaluator):
            evaluator.state.output = torch.cat(evaluator.state.y_preds).cpu()

        evaluator.run(loader)

        if one_hot:
            dataset_predictions = to_onehot(torch.argmax(evaluator.state.output, dim=1),
                                            num_classes=self._num_classes).float()
        else:
            dataset_predictions = F.softmax(evaluator.state.output, dim=1)

        return dataset_predictions

    def _get_labels(self, model, data):
        evaluator = create_supervised_evaluator(model, device=self._device,
                                                output_transform=lambda x, y, y_pred:
                                                torch.argmax(y_pred, dim=1))
        evaluator.logger = self._logger
        evaluator.state.labels = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def append_stolen_labels(evaluator):
            evaluator.state.labels.append(evaluator.state.output)

        @evaluator.on(Events.COMPLETED)
        def append_stolen_labels(evaluator):
            evaluator.state.output = torch.cat(evaluator.state.labels).cpu()

        ProgressBar().attach(evaluator)

        loader = DataLoader(data, pin_memory=True, num_workers=4, batch_size=256)
        evaluator.run(loader)

        return evaluator.state.output

    def _evaluate_model(self, model, data, labels=True):
        if labels:
            def output_tranform(x, y, y_pred):
                return y_pred, y
        else:
            def output_tranform(x, y, y_pred):
                return y_pred, torch.argmax(y, dim=1)

        metrics = {
            "macro_accuracy": MacroAccuracy(self._num_classes),
            "f1beta-score": Fbeta(beta=1)
        }
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=self._device,
                                                output_transform=output_tranform)

        ProgressBar().attach(evaluator)

        loader = DataLoader(dataset=data, batch_size=256, num_workers=4, pin_memory=True)
        evaluator.run(loader)

        return evaluator.state.metrics["macro_accuracy"], evaluator.state.metrics["f1beta-score"]

    def _add_ignite_events(self, trainer, evaluator, eval_loader, iteration):
        # Evaluator events
        def score_function(engine):
            val_macro_acc = engine.state.metrics["macro_accuracy"]
            return val_macro_acc

        early_stop = EarlyStopping(patience=self._test_config.early_stop_tolerance,
                                   score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop)

        to_save = {'copycat_model': self._substitute_model}
        checkpoint_handler = Checkpoint(to_save, DiskSaver(self._save_loc, require_empty=False),
                                        filename_prefix="best", score_function=score_function,
                                        score_name="f1beta-score",
                                        filename_pattern="{filename_prefix}_{name}_({score_name}="
                                                         "{score}).{ext}",
                                        global_step_transform=global_step_from_engine(trainer))
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        # Trainer events
        RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")
        ProgressBar().attach(trainer, ["avg_loss"])

        @trainer.on(Events.EPOCH_COMPLETED(every=self._test_config.evaluation_frequency))
        def log_validation_results(trainer):
            evaluator.run(eval_loader)
            metrics = evaluator.state.metrics
            trainer.logger.info("Validation results - Iteration: {} Epoch: {}  Macro-averaged "
                                "accuracy: {:.1f}% F1-score: {:.3f}"
                                .format(iteration, trainer.state.epoch,
                                        100 * metrics["macro_accuracy"],
                                        metrics["f1beta-score"]))

        return checkpoint_handler

    def _train_substitute_model(self, query_dataset, iteration):
        # Prepare trainer
        optimizer = torch.optim.Adam(self._substitute_model.parameters())
        loss_function = nn.MSELoss()
        trainer = create_supervised_trainer(self._substitute_model, optimizer, loss_function,
                                            device=self._device)
        trainer.logger = self._logger

        # Prepare evaluator
        def output_tranform(x, y, y_pred):
            return y_pred, torch.argmax(y, dim=1)

        metrics = {
            "macro_accuracy": MacroAccuracy(self._num_classes),
            "f1beta-score": Fbeta(beta=1)
        }
        evaluator = create_supervised_evaluator(self._substitute_model, metrics=metrics,
                                                device=self._device,
                                                output_transform=output_tranform)
        evaluator.logger = self._logger

        eval_loader = DataLoader(dataset=self._validation_dataset_predicted, batch_size=256,
                                 pin_memory=True, num_workers=4)
        checkpoint_handler = self._add_ignite_events(trainer, evaluator, eval_loader, iteration)

        # Start trainer
        train_loader = DataLoader(dataset=query_dataset, shuffle=True, batch_size=128,
                                  pin_memory=True, num_workers=4)
        trainer.run(train_loader, max_epochs=self._config.training_epochs)

        # Load best model and remove the file
        self._logger.info("Loading best model")
        to_load = {'copycat_model': self._substitute_model}
        checkpoint_fp = self._save_loc + '/' + checkpoint_handler.last_checkpoint
        checkpoint = torch.load(checkpoint_fp)
        Checkpoint.load_objects(to_load, checkpoint)
        Path.unlink(Path(checkpoint_fp))

        return

    def _entropy_strategy(self, k, idx, remaining_set_predictions):
        scores = {}
        for sample, prob_dist in zip(idx, remaining_set_predictions):
            log_probs = prob_dist * torch.log2(prob_dist)
            raw_entropy = 0 - torch.sum(log_probs)

            normalized_entropy = raw_entropy / math.log2(prob_dist.numel())

            scores[sample] = normalized_entropy.item()

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return list(scores.keys())[:k]

    def _random_strategy(self, k, idx):
        idx_copy = np.copy(idx)
        np.random.shuffle(idx_copy)

        return list(idx_copy[:k])

    def _kcenter_strategy(self, k, idx, remaining_samples_predictions, query_sets_predictions):
        loader = DataLoader(remaining_samples_predictions, batch_size=256, num_workers=4,
                            pin_memory=True)
        kc = KCenter(self._device)

        def batch_step(engine, batch):
            return kc(batch, query_sets_predictions)

        evaluator = Engine(batch_step)
        ProgressBar().attach(evaluator)

        evaluator.state.min_max_values = []
        evaluator.state.min_max_idx = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def append_most_distant(evaluator):
            evaluator.state.min_max_values.append(evaluator.state.output[0])
            evaluator.state.min_max_idx.append(evaluator.state.output[1])

        selected_points = []
        for i in range(k):
            evaluator.state.min_max_values = []
            evaluator.state.min_max_idx = []

            evaluator.run(loader)

            batch_id = np.argmax(evaluator.state.min_max_values)
            sample_id = batch_id * 256 + evaluator.state.min_max_idx[batch_id.item()]
            query_sets_predictions = torch.from_numpy(np.vstack(
                [query_sets_predictions.cpu(), remaining_samples_predictions[sample_id]]))

            selected_points.append(sample_id)

        return [idx[point] for point in selected_points]

    def _deepfool_strategy(self, k, idx, remaining_samples):
        self._substitute_model.eval()
        loader = DataLoader(remaining_samples, batch_size=256, num_workers=4, pin_memory=True)

        df = Dfal(self._substitute_model, self._num_classes)

        def batch_step(engine, batch):
            return df(batch[0])

        evaluator = Engine(batch_step)
        ProgressBar().attach(evaluator)

        evaluator.state.scores_values = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def append_scores(evaluator):
            evaluator.state.scores_values.extend(evaluator.state.output)

        evaluator.run(loader)
        scores_values = evaluator.state.scores_values

        scores = dict(zip(idx, scores_values))
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return list(scores.keys())[:k]

    def _select_samples(self, idx, remaining_samples, remaining_samples_predictions, query_sets):
        if self._config.selection_strategy == "entropy":
            selected_samples = self._entropy_strategy(self._config.k, idx,
                                                      remaining_samples_predictions)
        elif self._config.selection_strategy == "random":
            selected_samples = self._random_strategy(self._config.k, idx)
        elif self._config.selection_strategy == "k-center":
            # Get initial centers
            query_sets_predictions = self._get_predictions(self._substitute_model, query_sets)
            selected_samples = self._kcenter_strategy(self._config.k, idx,
                                                      remaining_samples_predictions,
                                                      query_sets_predictions)
        elif self._config.selection_strategy == "dfal":
            selected_samples = self._deepfool_strategy(self._config.k, idx, remaining_samples)
        else:
            self._logger.warning("Selection strategy must be one of {entropy, random, k-center}")
            raise ValueError

        return selected_samples

    def run(self):
        self._logger.info("########### Starting ActiveThief attack ###########")
        # Get budget of the attack
        budget = self._config.initial_seed_size + len(self._validation_dataset_original) + \
                 (self._config.iterations * self._config.k)
        self._logger.info("ActiveThief's attack budget: {}".format(budget))

        available_samples = set(range(len(self._thief_dataset)))
        query_sets = []
        query_sets_predictions = []

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        validation_predictions = self._get_predictions(self._secret_model,
                                                       self._validation_dataset_original,
                                                       self._config.one_hot_output)
        self._validation_dataset_predicted = CustomLabelDataset(self._validation_dataset_original,
                                                                validation_predictions)

        validation_label_counts = dict(list(enumerate([0] * self._num_classes)))
        for class_id in torch.argmax(validation_predictions, dim=1):
            validation_label_counts[class_id.item()] += 1

        self._logger.info("Validation dataset labels distribution: {}".format(
            validation_label_counts))

        # Step 1: attacker picks random subset of initial seed samples
        self._logger.info("Preparing initial random query set")
        idx = np.random.choice(np.arange(len(available_samples)),
                               size=self._config.initial_seed_size, replace=False)
        available_samples -= set(idx)

        query_sets.append(torch.utils.data.Subset(self._thief_dataset, idx))
        query_set_predictions = self._get_predictions(self._secret_model, query_sets[0],
                                                      self._config.one_hot_output)
        query_sets_predictions.append(query_set_predictions)

        # Get secret model predicted labels for test dataset
        self._logger.info("Getting secret model's labels for test dataset")
        true_test_labels = self._get_labels(self._secret_model, self._test_dataset).numpy()

        self._logger.info("Number of test samples: {}".format(len(true_test_labels)))

        # Save substitute model state_dict for retraining from scratch
        copy_state_dict = self._substitute_model.state_dict()

        for iteration in range(1, self._config.iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(iteration))

            # Get metrics from secret model and substitute model
            self._logger.info("Test dataset metrics")
            self._logger.info("Secret model accuracy: {:.3f} F1-score: {:.3f}".format(
                *self._evaluate_model(self._secret_model, self._test_dataset)))
            self._logger.info("Substitute model accuracy: {:.3f} F1-score: {:.3f}".format(
                *self._evaluate_model(self._substitute_model, self._test_dataset)))

            # Reset substitute model
            self._substitute_model.load_state_dict(copy_state_dict)

            # Step 3: The substitute model is trained with union of all the labeled queried sets
            self._logger.info("Training substitute model with the query dataset")
            query_dataset = CustomLabelDataset(ConcatDataset(query_sets),
                                               torch.cat(query_sets_predictions))
            self._train_substitute_model(query_dataset, iteration)
            # Save the state dictionary of the best model this iteration
            torch.save(dict(state_dict=self._substitute_model.state_dict()), self._save_loc +
                       "/best_substitute_model_iteration={}".format(iteration))

            # Agreement score
            substitute_test_labels = self._get_labels(self._substitute_model,
                                                      self._test_dataset).numpy()
            agreement_count = np.sum(true_test_labels == substitute_test_labels)
            self._logger.info("Agreement count: {}".format(agreement_count))
            self._logger.info("Test agreement between secret and substitute model on true test "
                              "dataset {:.1f}"
                              .format(100 * (agreement_count / len(true_test_labels))))

            # Step 4: Approximate labels are obtained for remaining samples using the substitute
            self._logger.info("Getting substitute's predictions for the rest of the thief dataset")
            idx = sorted(list(available_samples))
            remaining_samples = torch.utils.data.Subset(self._thief_dataset, idx)
            remaining_samples_predictions = self._get_predictions(self._substitute_model,
                                                                  remaining_samples)

            # Step 5: An active learning subset selection strategy is used to select set of k
            # samples
            self._logger.info("Selecting {} samples using the {} strategy from the remaining thief "
                              "dataset".format(self._config.k, self._config.selection_strategy))
            selected_samples = self._select_samples(idx, remaining_samples,
                                                    remaining_samples_predictions,
                                                    ConcatDataset(query_sets))
            available_samples -= set(selected_samples)
            query_sets.append(torch.utils.data.Subset(self._thief_dataset, selected_samples))

            # Step 2: Attacker queries current picked samples to secret model for labeling
            self._logger.info("Getting predictions for the current query set from the secret model")
            query_set_predictions = self._get_predictions(self._secret_model, query_sets[iteration],
                                                          self._config.one_hot_output)
            query_sets_predictions.append(query_set_predictions)