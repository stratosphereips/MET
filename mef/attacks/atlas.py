import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomDataset, CustomLabelDataset, \
    NoYDataset
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer


class SimpleUncertaintyPredictor(pl.LightningModule):

    def __init__(self, feature_vec_size):
        super().__init__()
        self._model = nn.Sequential(nn.Linear(feature_vec_size, 2),
                                    nn.ReLU(inplace=True))

    @auto_move_data
    def forward(self, feature_vec):
        return self._model(feature_vec)


class AtlasThief(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 iterations=10, output_type="softmax", budget=20000,
                 training_epochs=1000, early_stop_tolerance=100,
                 evaluation_frequency=2, batch_size=64,
                 save_loc="./cache/AtlasThief", gpus=0, seed=None,
                 deterministic=True, debug=False, precision=32):
        optimizer = torch.optim.Adam(substitute_model.parameters(),
                                     weight_decay=1e-3)
        loss = soft_cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss,
                         training_epochs=training_epochs,
                         early_stop_tolerance=early_stop_tolerance,
                         evaluation_frequency=evaluation_frequency,
                         batch_size=batch_size, num_classes=num_classes,
                         save_loc=save_loc, gpus=gpus, seed=seed,
                         deterministic=deterministic, debug=debug,
                         precision=precision)

        # AtlasThief's specific attributes
        self._iterations = iterations
        self._output_type = output_type
        self._budget = budget
        # Values from paper
        self._init_seed_size = int(self._budget * 0.1)
        self._val_size = int(self._budget * 0.2)
        self._k = (self._budget - self._val_size - self._init_seed_size) // \
                  self._iterations

        if self._k <= 0:
            self._logger.error("ActiveThief's per iteration selection must "
                               "be bigger than 0!")
            raise ValueError()

    def _get_train_set(self, val_set):
        self._logger.info(
                "Getting correct/incorrect labels and hidden layer output "
                "for validation set from substitute model")
        self._substitute_model.eval()

        val_loader = DataLoader(val_set, batch_size=self._batch_size,
                                pin_memory=True, num_workers=4)

        hidden_layer_output = []
        corr_incorr_labels = []  # 0-incorrect, 1-correct
        for x, y in tqdm(val_loader,
                         desc="Getting validation set labels and hidden "
                              "layer output"):
            logits, hidden_layer = self._substitute_model(x, True)

            preds = logits.argmax(dim=1).cpu()
            targets = torch.argmax(y, dim=1)

            corr_incorr_labels.append((preds == targets).long())
            hidden_layer_output.append(hidden_layer.detach().cpu())

        return torch.cat(hidden_layer_output), torch.cat(corr_incorr_labels)

    def _train_new_output_layer(self, train_set):
        self._logger.info("Training a new model to predict whether "
                          "validation items were correct or incorrect")

        deterministic = self._trainer_kwargs["deterministic"]
        debug = False
        precision = self._trainer_kwargs["precision"]
        trainer = get_trainer(save_loc=self._save_loc + "/correct-model",
                              deterministic=deterministic, debug=debug,
                              validation=False, precision=precision)

        correct_model = SimpleUncertaintyPredictor(train_set[0][0].shape[0])
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(correct_model.parameters(), lr=0.01)
        mef_model = MefModule(correct_model, 2, optimizer, loss)

        train_loader = DataLoader(train_set, batch_size=self._batch_size,
                                  shuffle=True, num_workers=4)
        trainer.fit(mef_model, train_loader)

        return correct_model

    def _get_atl_sample(self, k, correct_model, idxs_rest, data_rest):
        self._logger.info(
                "Predict whether unlabeled data is correct/incorrect")
        _, hidden_layer_outputs = self._get_predictions(self._substitute_model,
                                                        data_rest,
                                                        return_all_layers=True)

        hidden_layer_data = NoYDataset(hidden_layer_outputs)
        y_preds = self._get_predictions(correct_model, hidden_layer_data)

        probs_incorrect = y_preds.data[:, 1]
        best = torch.topk(probs_incorrect, k)

        return idxs_rest[best.indices], hidden_layer_outputs[best.indices]

    def _atlas_strategy(self, idxs_rest, val_set):
        idxs_rest_orig = idxs_rest.copy()

        new_train_data, new_train_labels = self._get_train_set(val_set)
        train_data = [new_train_data]
        train_labels = [new_train_labels]
        selected_points = []
        for _ in tqdm(range(self._k), desc="Selecting best points"):
            train_set = CustomDataset(torch.cat(train_data),
                                      torch.cat(train_labels))
            data_rest = Subset(self._thief_dataset, idxs_rest)
            correct_model = self._train_new_output_layer(train_set)
            idxs_best, new_train_data = self._get_atl_sample(10, correct_model,
                                                             idxs_rest,
                                                             data_rest)
            idxs_rest = np.setdiff1d(idxs_rest, idxs_best)
            selected_points.append(idxs_best)

            train_data.append(new_train_data)
            train_labels.append(torch.ones(10).long())

        return np.concatenate(selected_points)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info(
                "########### Starting AtlasThief attack ###########")
        # Get budget of the attack
        self._logger.info(
                "AtlasThief's attack budget: {}".format(self._budget))

        idxs_rest = np.arange(len(self._thief_dataset))

        # Prepare validation set
        self._logger.info("Preparing validation dataset")
        idxs_val = np.random.permutation(idxs_rest)[:self._val_size]
        idxs_rest = np.setdiff1d(idxs_rest, idxs_val)
        val_set = Subset(self._thief_dataset, idxs_val)
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

        idxs_query = np.random.permutation(idxs_rest)[:self._init_seed_size]
        ixds_rest = np.setdiff1d(idxs_rest, idxs_query)
        query_set = Subset(self._thief_dataset, idxs_query)
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
        vict_test_acc = self._test_model(self._victim_model, self._test_set)

        # Save substitute model state_dict for retraining from scratch
        sub_orig_state_dict = self._substitute_model.state_dict()
        optim_orig_state_dict = self._optimizer.state_dict()

        for it in range(1, self._iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it))

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
            self._train_model(self._substitute_model, self._optimizer,
                              train_set, val_set, it)

            self._get_aggreement_score()

            # Step 4: An Atlas subset selection strategy is used
            # to select set of k samples
            self._logger.info(
                    "Selecting {} samples from the remaining thief dataset"
                        .format(self._k))
            idxs_query = self._atlas_strategy(ixds_rest, val_set)
            idxs_rest = np.setdiff1d(idxs_rest, idxs_query)
            query_set = Subset(self._thief_dataset, idxs_query)

            # Step 2: Attacker queries current picked samples to secret
            # model for labeling
            self._logger.info("Getting predictions for the current query set "
                              "from the victim model")
            y_query = self._get_predictions(self._victim_model, query_set,
                                            self._output_type)
            query_sets.append(CustomLabelDataset(query_set, y_query))

        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()

        return
