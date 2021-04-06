import pickle
from argparse import ArgumentParser
from collections import defaultdict as dd
from dataclasses import dataclass
from typing import Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from mef.attacks.base import AttackBase
from mef.utils.pytorch.datasets import CustomLabelDataset, split_dataset
from mef.utils.pytorch.functional import get_prob_vector, soft_cross_entropy
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.settings import AttackSettings


@dataclass
class KnockOffSettings(AttackSettings):
    iterations: int
    output_type: str
    budget: int
    init_seed_size: int
    val_size: int
    samples_per_iteration: int
    save_samples: bool

    def __init__(
        self, sampling_strategy: str, reward_type: str, budget: int, samples_per_iteration: int, save_samples: bool
    ):
        self.sampling_strategy = sampling_strategy.lower()
        self.reward_type = reward_type.lower()
        self.budget = budget
        self.samples_per_iteration = samples_per_iteration
        self.iterations = self.budget // self.samples_per_iteration
        self.save_samples = save_samples

        # Check configuration
        if self.sampling_strategy not in ["random", "adaptive"]:
            raise ValueError(
                "Knockoff's sampling strategy must be one of {random, adaptive}"
            )

        if self.reward_type not in ["cert", "div", "loss", "all"]:
            raise ValueError(
                "Knockoff's reward type must be one of {cert, div, loss, all}"
            )


class KnockOff(AttackBase):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        online_optimizer: torch.optim.Optimizer,
        sampling_strategy: str = "adaptive",
        reward_type: str = "cert",
        budget: int = 20000,
        samples_per_iteration: int = 4,
        save_samples: bool = False,
    ):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = KnockOffSettings(
            sampling_strategy, reward_type, budget, samples_per_iteration, save_samples
        )

        # KnockOff's specific attributes
        self._online_optimizer = online_optimizer
        self._online_loss = self._substitute_model._loss

        self._selected_samples = dd(list)
        self._y = None
        self._actions = None
        self._num_actions = None
        self._y_avg = None
        self._reward_avg = None
        self._reward_var = None

    @classmethod
    def _get_attack_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(description="KnockOff attack")
        parser.add_argument(
            "--sampling_strategy",
            default="adaptive",
            type=str,
            help="KnockOff-Nets sampling strategy can "
            "be one of {random, adaptive} ("
            "Default: adaptive)",
        )
        parser.add_argument(
            "--reward_type",
            default="all",
            type=str,
            help="Type of reward for adaptive strategy, "
            "can be one of {cert, div, loss, all} "
            "(Default: all)",
        )
        parser.add_argument(
            "--budget",
            default=20000,
            type=int,
            help="Size of the budget (Default: 20000)",
        )
        parser.add_argument(
            "--samples_per_iteration",
            default=4,
            type=int,
            help="Number of samples selected in each iteration (Default: 4).",
        )
        parser.add_argument(
            "--idxs",
            action="store_true",
            help="Whether to save idxs of samples selected "
            "during the attacks. (Default: False)",
        )

        return parser

    def _random_strategy(self) -> CustomLabelDataset:
        self._logger.info(
            "Selecting random sample from thief dataset of "
            "size {}".format(self.attack_settings.budget)
        )
        idx_x = np.arange(len(self._thief_dataset))
        idx_sampled = np.random.permutation(idx_x)[: self.attack_settings.budget]
        selected_data = Subset(self._thief_dataset, idx_sampled)

        self._logger.info("Getting fake labels from victim model")
        y = self._get_predictions(self._victim_model, selected_data)

        if self.attack_settings.save_samples:
            self._selected_samples["idxs"].extend(idx_sampled)
            self._selected_samples["labels"].append(y)

        return CustomLabelDataset(selected_data, y)

    def _sample_data(self, action: int) -> Union[Subset, bool]:
        idx_action = np.where(self._y == action)[0]
        if len(idx_action) == 0:
            return False

        idx_sampled = np.random.permutation(idx_action)[: self.attack_settings.samples_per_iteration]

        # Mark selected samples from possible samples for selection
        self._y[idx_sampled] = -1

        if self.attack_settings.save_samples:
            self._selected_samples["idxs"].extend(idx_sampled)

        return Subset(self._thief_dataset, idx_sampled)

    def _online_train(self, data: Dataset) -> None:
        self._substitute_model.train()
        loader = DataLoader(
            dataset=data,
            pin_memory=self.base_settings.gpu,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        for x, y_output in tqdm(loader, desc="Online training"):
            if self.base_settings.gpu:
                x = x.cuda()
                y_output = y_output.cuda()

            self._online_optimizer.zero_grad()

            logits = self._substitute_model(x)[0]
            loss = self._online_loss(logits, y_output)
            loss.backward()
            self._online_optimizer.step()

        return

    def _reward_cert(self, y_output: torch.Tensor) -> np.ndarray:
        largests = torch.topk(y_output, 2)[0]
        reward = torch.mean((largests[:, 0] - largests[:, 1]))

        return reward.numpy()

    def _reward_div(self, y: torch.Tensor, n: int) -> np.ndarray:
        # First update y_avg
        self._y_avg = self._y_avg + (1.0 / n) * (y.mean(dim=0) - self._y_avg)

        # Then compute reward
        reward = torch.mean(torch.sum(np.maximum(0, y - self._y_avg), dim=-1))

        return reward.numpy()

    def _reward_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        # Compute reward
        reward = self._online_loss(y_hat, y)

        return reward.numpy()

    def _reward_all(self, y_hat: torch.Tensor, y: torch.Tensor, n: int) -> np.ndarray:
        # Source: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/extraction/knockoff_nets.py
        reward_cert = self._reward_cert(y)
        reward_div = self._reward_div(y, n)
        reward_loss = self._reward_loss(y, y_hat)
        reward = [reward_cert, reward_div, reward_loss]
        self._reward_avg = self._reward_avg + (1.0 / n) * (reward - self._reward_avg)
        self._reward_var = self._reward_var + (1.0 / n) * (
            (reward - self._reward_avg) ** 2 - self._reward_var
        )

        # Normalize rewards
        if n > 1:
            reward = (reward - self._reward_avg) / np.sqrt(self._reward_var)
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        return np.mean(reward)

    def _reward(
        self, y_hat: torch.Tensor, y: torch.Tensor, iteration: int
    ) -> np.ndarray:
        reward_type = self.attack_settings.reward_type
        if reward_type == "cert":
            return self._reward_cert(y)
        elif reward_type == "div":
            return self._reward_div(y, iteration)
        elif reward_type == "loss":
            return self._reward_loss(y_hat, y)
        else:
            return self._reward_all(y_hat, y, iteration)

    def _adaptive_strategy(self) -> ConcatDataset:
        # Get substitute dataset labels
        if hasattr(self._thief_dataset, "targets"):
            self._y = self._thief_dataset.targets
        else:
            loader = DataLoader(
                dataset=self._thief_dataset,
                pin_memory=self.base_settings.gpu,
                num_workers=self.base_settings.num_workers,
                batch_size=self.base_settings.batch_size,
            )
            self._y = []
            for _, labels in tqdm(
                loader, desc="Collecting substitute datasets' " "labels"
            ):
                self._y.append(labels)
            self._y = torch.cat(self._y)

        self._y = np.array(self._y, dtype=np.uint32)
        self._actions = np.unique(self._y)
        self._num_actions = len(self._actions)

        # We need to keep an average version of the victim output
        if (
            self.attack_settings.reward_type == "div"
            or self.attack_settings.reward_type == "all"
        ):
            self._y_avg = torch.zeros(self._victim_model.num_classes)

        # We need to keep an average and variance version of rewards
        if self.attack_settings.reward_type == "all":
            self._reward_avg = np.zeros(3)
            self._reward_var = np.zeros(3)

        # Implement the bandit gradients algorithm
        h_func = torch.zeros(self._num_actions)
        learning_rate = torch.zeros(self._num_actions)
        probs = np.ones(self._num_actions) / self._num_actions
        query_sets = []

        avg_reward = np.array(0.0)
        for it in tqdm(
            range(1, self.attack_settings.iterations + 1), desc="Selecting data samples"
        ):
            self._logger.info("---------- Iteration: {} ----------".format(it))
            # Sample actions until we get data to continue the attack
            while True:
                # Sample an action
                action_idx = np.random.choice(np.arange(0, self._num_actions), p=probs)
                action = self._actions[action_idx]
                self._logger.info("Action {} selected".format(action))

                # Select sample to attack
                self._logger.info("Selecting sample for attack")
                sampled_data = self._sample_data(action)

                if sampled_data:
                    break
                else:
                    self._logger.info(
                        "Action has no more samples, removing it from pool of actions"
                    )
                    # Action has no more samples, we remove it from selection
                    action_prob = probs[action]
                    probs = np.delete(probs, action)
                    h_func = np.delete(h_func, action)
                    self._num_actions -= 1
                    self._y[self._y > action] -= 1
                    # We need to make sure that probs still add up to 1
                    probs += np.float32((action_prob / self._num_actions))

            # Query the victim model
            self._logger.info("Getting victim predictions on sampled data")
            y = self._get_predictions(self._victim_model, sampled_data)

            if self.attack_settings.save_samples:
                self._selected_samples["labels"].append(y)

            # Train the thieved classifier
            query_set = CustomLabelDataset(sampled_data, y)
            query_sets.append(query_set)
            self._online_train(query_set)

            # Test new labels
            self._logger.info("Getting substitute predictions on sampled data")
            y_hat = self._get_predictions(self._substitute_model, sampled_data)
            y_hat = get_prob_vector(y_hat)

            # Compute rewards
            self._logger.info("Computing rewards")
            reward = self._reward(y_hat, y, it)
            self._logger.info("Reward: {}".format(reward))
            avg_reward = avg_reward + (1.0 / it) * (reward - avg_reward)
            self._logger.info("Average Reward: {}".format(avg_reward))

            # Update learning rate
            learning_rate[action] += 1

            # Update H function (action preferences)
            for a in range(self._num_actions):
                if a != action:
                    h_func[a] = h_func[a] - (1.0 / learning_rate[action]) * (
                        (reward - avg_reward) * probs[a]
                    )
                else:
                    h_func[a] = h_func[a] + (1.0 / learning_rate[action]) * (
                        (reward - avg_reward) * (1 - probs[a])
                    )

            # Update probs
            probs = F.softmax(h_func, dim=0).numpy()

        return ConcatDataset(query_sets)

    def _check_args(self, sub_data: Dataset, test_set: Dataset) -> None:
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's " "dataset.")
            raise TypeError()
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._thief_dataset = sub_data
        self._test_set = test_set

        return

    def _run(self, sub_data: Dataset, test_set: Dataset) -> None:
        self._check_args(sub_data, test_set)
        self._logger.info("########### Starting KnockOff attack ###########")
        self._logger.info(
            "KnockOff's attack budget: {}".format(self.attack_settings.budget)
        )

        if self.attack_settings.sampling_strategy == "random":
            self._logger.info("Starting random sampling strategy")
            transfer_data = self._random_strategy()
        else:
            original_state_dict = self._substitute_model.model.state_dict()
            self._logger.info(
                "Starting adaptive sampling strategy with {} reward type".format(
                    self.attack_settings.reward_type
                )
            )
            transfer_data = self._adaptive_strategy()
            self._substitute_model.model.load_state_dict(original_state_dict)

        if self.attack_settings.save_samples:
            idxs_filepath = self.base_settings.save_loc.joinpath("selected_samples.pl")
            self._selected_samples["labels"] = torch.cat(
                self._selected_samples["labels"]
            )
            with open(idxs_filepath, "wb") as f:
                pickle.dump(self._selected_samples, f)

        train_set = transfer_data
        val_set = None
        if self.trainer_settings.evaluation_frequency is not None:
            train_set, val_set = split_dataset(train_set, 0.2)

        self._logger.info("Offline training of substitute model")
        self._train_substitute_model(train_set, val_set)

        return
