# Based on https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob
# /main/art/attacks/extraction/knockoff_nets.py
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from mef.attacks.base import AttackSettings, Base
from mef.utils.pytorch.datasets import CustomLabelDataset, MefDataset
from mef.utils.pytorch.functional import soft_cross_entropy


@dataclass
class KnockOffSettings(AttackSettings):
    iterations: int
    output_type: str
    budget: int
    init_seed_size: int
    val_size: int
    k: int

    def __init__(self,
                 sampling_strategy: str,
                 reward_type: str,
                 output_type: str,
                 budget: int):
        self.sampling_strategy = sampling_strategy.lower()
        self.reward_type = reward_type.lower()
        self.output_type = output_type.lower()
        self.budget = budget
        self.k = 4
        self.iterations = self.budget // self.k

        self.init_seed_size = int(self.budget * 0.1)
        self.val_size = int(self.budget * 0.2)
        self.k = (self.budget - self.val_size - self.init_seed_size) // \
                 self.iterations

        # Check configuration
        if self.sampling_strategy not in ["random", "adaptive"]:
            raise ValueError(
                    "Knockoff's sampling strategy must be one of {random, "
                    "adaptive}")

        if self.reward_type not in ["cert", "div", "loss", "all"]:
            raise ValueError(
                    "Knockoff's reward type must be one of {cert, div, loss, "
                    "all}")


class KnockOff(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 sampling_strategy="adaptive", reward_type="cert",
                 output_type="softmax", budget=10000):
        optimizer = torch.optim.SGD(substitute_model.parameters(), lr=0.01,
                                    momentum=0.5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=60)
        loss = soft_cross_entropy

        super().__init__(victim_model, substitute_model, optimizer,
                         loss, lr_scheduler)
        self.attack_settings = KnockOffSettings(sampling_strategy,
                                                reward_type, output_type,
                                                budget)
        self.trainer_settings.validation = False
        self.data_settings._num_classes = num_classes

        # KnockOff's specific attributes
        self._online_optimizer = torch.optim.SGD(
                self._substitute_model.parameters(), lr=0.0005, momentum=0.5)

        self._selected_action = []
        self._selected_idxs = []
        self._num_actions = None
        self._y_avg = None
        self._reward_avg = None
        self._reward_var = None

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="KnockOff attack")
        parser.add_argument("--sampling_strategy", default="adaptive",
                            type=str,
                            help="KnockOff-Nets sampling strategy can "
                                 "be one of {random, adaptive} ("
                                 "Default: adaptive)")
        parser.add_argument("--reward_type", default="all", type=str,
                            help="Type of reward for adaptive strategy, "
                                 "can be one of {cert, div, loss, all} "
                                 "(Default: all)")
        parser.add_argument("--output_type", default="softmax", type=str,
                            help="Type of output from victim model {softmax, "
                                 "logits, one_hot, labels} (Default: softmax)")
        parser.add_argument("--budget", default=20000, type=int,
                            help="Size of the budget (Default: 20000)")
        parser.add_argument("--substitute_train_epochs", default=100, type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 100)")
        parser.add_argument("--patience", default=10, type=int,
                            help="Number of epochs without improvement for "
                                 "early stop (Default: 10)")
        cls._add_base_args(parser)

        return

    def _random_strategy(self):
        self._logger.info("Selecting random sample from thief dataset of "
                          "size {}".format(self.attack_settings.budget))
        idx_x = np.arange(len(self._thief_dataset))
        idx_selected = np.random.permutation(idx_x)[
                       :self.attack_settings.budget]
        selected_data = Subset(self._thief_dataset, idx_selected)

        self._logger.info("Getting fake labels from victim model")
        y_output = self._get_predictions(self._victim_model, selected_data,
                                         self.attack_settings.output_type)

        return CustomLabelDataset(selected_data, y_output)

    def _sample_data(self, action):
        if isinstance(self._thief_dataset, Subset):
            idx_sub = np.array(self._thief_dataset.indices)
            y = np.array(self._thief_dataset.dataset.targets)
            y = y[idx_sub]
        else:
            y = np.array(self._thief_dataset.targets)

        idx_action = np.where(y == action)[0]
        idx_sampled = np.random.permutation(idx_action)[
                      :self.attack_settings.k]
        self._selected_idxs.append(idx_sampled)

        return Subset(self._thief_dataset, idx_sampled)

    def _online_train(self, data):
        self._substitute_model.train()
        data = MefDataset(data, self.data_settings.batch_size)
        loader = data.generic_dataloader()

        for x, y_output in tqdm(loader, desc="Online training"):
            if self.base_settings.gpus is not None:
                x = x.cuda()
                y_output = y_output.cuda()

            self._online_optimizer.zero_grad()

            y_hat = self._substitute_model(x)
            loss = self._loss(y_hat, y_output)
            loss.backward()
            self._online_optimizer.step()

        return

    def _reward_cert(self, y_output):
        """
        Compute `cert` reward value.
        """
        largests = torch.topk(y_output, 2)[0]
        reward = torch.mean((largests[:, 0] - largests[:, 1]))

        return reward.numpy()

    def _reward_div(self, y_output, n):
        """
        Compute `div` reward value.
        """
        # First update y_avg
        self._y_avg = self._y_avg + (1.0 / n) * \
                      ((y_output.sum(dim=0) / self.attack_settings.k) -
                       self._y_avg)

        # Then compute reward
        reward = torch.mean(torch.sum(np.maximum(0, y_output - self._y_avg),
                                      dim=1))

        return reward.numpy()

    def _reward_loss(self, y_output, y_hat):
        """
        Compute `loss` reward value.
        """

        # Compute reward
        reward = torch.mean(torch.sum(-y_output * F.log_softmax(y_hat, dim=1),
                                      dim=1))

        return reward.numpy()

    def _reward_all(self, y_output, y_hat, n):
        """
        Compute `all` reward value.
        """
        reward_cert = self._reward_cert(y_output)
        reward_div = self._reward_div(y_output, n)
        reward_loss = self._reward_loss(y_output, y_hat)
        reward = [reward_cert, reward_div, reward_loss]
        self._reward_avg = self._reward_avg + (1.0 / n) * \
                           (reward - self._reward_avg)
        self._reward_var = self._reward_var + (1.0 / n) * \
                           ((reward - self._reward_avg) ** 2 -
                            self._reward_var)

        # Normalize rewards
        if n > 1:
            reward = (reward - self._reward_avg) / np.sqrt(self._reward_var)
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        return np.mean(reward)

    def _reward(self, y_output, y_hat, iteration):
        reward_type = self.attack_settings.reward_type
        if reward_type == "cert":
            return self._reward_cert(y_output)
        elif reward_type == "div":
            return self._reward_div(y_output, iteration)
        elif reward_type == "loss":
            return self._reward_loss(y_output, y_hat)
        else:
            return self._reward_all(y_output, y_hat, iteration)

    def _adaptive_strategy(self):
        # Number of actions
        if isinstance(self._thief_dataset, Subset):
            self._num_actions = len(
                    np.unique(self._thief_dataset.dataset.targets))
        else:
            self._num_actions = len(
                    np.unique(self._thief_dataset.targets))

        # We need to keep an average version of the victim output
        if self.attack_settings.reward_type == "div" or \
                self.attack_settings.reward_type == "all":
            self._y_avg = torch.zeros(self.data_settings._num_classes)

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
        for it in range(1, self.attack_settings.iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it))
            # Sample an action
            action = np.random.choice(np.arange(0, self._num_actions), p=probs)
            self._selected_action.append(action)
            self._logger.info("Action {} selected".format(action))

            # Select sample to attack
            self._logger.info("Selecting sample for attack")
            sampled_data = self._sample_data(action)

            # Query the victim model
            self._logger.info("Getting victim predictions on sampled data")
            y_output = self._get_predictions(self._victim_model, sampled_data,
                                             self.attack_settings.output_type)

            # Train the thieved classifier
            query_set = CustomLabelDataset(sampled_data, y_output)
            query_sets.append(query_set)
            self._online_train(query_set)

            # Test new labels
            self._logger.info("Getting substitute predictions on sampled data")
            y_hat = self._get_predictions(self._substitute_model, sampled_data)

            # Compute rewards
            self._logger.info("Computing rewards")
            reward = self._reward(y_output, y_hat, it)
            self._logger.info("Reward: {}".format(reward))
            avg_reward = avg_reward + (1.0 / it) * (reward - avg_reward)
            self._logger.info("Average Reward: {}".format(avg_reward))

            # Update learning rate
            learning_rate[action] += 1

            # Update H function (action preferences)
            for a in range(self._num_actions):
                if a != action:
                    h_func[a] = h_func[a] - (1.0 / learning_rate[action]) * \
                                ((reward - avg_reward) * probs[a])
                else:
                    h_func[a] = h_func[a] + (1.0 / learning_rate[action]) * \
                                ((reward - avg_reward) * (1 - probs[a]))

            # Update probs
            probs = F.softmax(h_func, dim=0).numpy()

        return ConcatDataset(query_sets)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info(
                "########### Starting KnockOff attack ###########")
        self._logger.info(
                "KnockOff's attack budget: {}".format(
                        self.attack_settings.budget))

        if self.attack_settings.sampling_strategy == "random":
            self._logger.info("Starting random sampling strategy")
            transfer_data = self._random_strategy()
        else:
            original_state_dict = self._substitute_model.state_dict()
            self._logger.info(
                    "Starting adaptive sampling strategy with {} reward type"
                        .format(self.attack_settings.reward_type))
            transfer_data = self._adaptive_strategy()
            self._substitute_model.load_state_dict(original_state_dict)

        self._logger.info("Offline training of substitute model")
        self._train_substitute_model(transfer_data)

        self._finalize_attack()

        return
