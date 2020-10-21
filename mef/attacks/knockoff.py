# Based on https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob
# /main/art/attacks/extraction/knockoff_nets.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomLabelDataset


def soft_cross_entropy(y_hat, y_output, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- y_output * F.log_softmax(y_hat, dim=1) *
                                    weights, dim=1))
    else:
        return torch.mean(torch.sum(- y_output * F.log_softmax(y_hat, dim=1),
                                    dim=1))


class KnockOff(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 sampling_strategy="adaptive", reward_type="cert",
                 output_type="softmax", budget=1000, training_epochs=100,
                 batch_size=64, save_loc="./cache/knockoff", gpus=0, seed=None,
                 deterministic=True, debug=False):
        optimizer = torch.optim.SGD(substitute_model.parameters(), lr=0.01,
                                    momentum=0.5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=60)
        train_loss = soft_cross_entropy
        test_loss = torch.nn.CrossEntropyLoss()

        super().__init__(victim_model, substitute_model, optimizer,
                         train_loss, test_loss, lr_scheduler, training_epochs,
                         batch_size=batch_size, num_classes=num_classes,
                         save_loc=save_loc, validation=False, gpus=gpus,
                         seed=seed, deterministic=deterministic, debug=debug)

        # KnockOff's specific attributes
        self._online_optimizer = torch.optim.SGD(
                self._substitute_model.parameters(), lr=0.0005, momentum=0.5)
        self._sampling_strategy = sampling_strategy
        self._reward_type = reward_type
        self._output_type = output_type
        self._budget = budget
        self._k = 4
        self._iterations = self._budget // self._k
        self._selected_action = []
        self._selected_idxs = []

        self._num_actions = None

        self._y_avg = None
        self._reward_avg = None
        self._reward_var = None

        # Check configuration
        if self._sampling_strategy not in ["random", "adaptive"]:
            self._logger.error(
                    "Knockoff's sampling strategy must be one of {random, "
                    "adaptive}")
            raise ValueError()

    def _random_strategy(self):
        self._logger.info("Selecting random sample from thief dataset of "
                          "size {}".format(self._budget))
        idx_x = np.arange(len(self._sub_dataset))
        idx_selected = np.random.permutation(idx_x)[:self._budget]
        selected_data = Subset(self._sub_dataset, idx_selected)

        self._logger.info("Getting fake labels from victim model")
        y_output = self._get_predictions(self._victim_model, selected_data,
                                         self._output_type)

        return CustomLabelDataset(selected_data, y_output)

    def _sample_data(self, action):
        if isinstance(self._sub_dataset, Subset):
            idx_sub = np.array(self._sub_dataset.indices)
            y = np.array(self._sub_dataset.dataset.targets)
            y = y[idx_sub]
        else:
            y = np.array(self._sub_dataset.targets)

        idx_action = np.where(y == action)[0]
        idx_sampled = np.random.permutation(idx_action)[:self._k]
        self._selected_idxs.append(idx_sampled)

        return Subset(self._sub_dataset, idx_sampled)

    def _online_train(self, data):
        self._substitute_model.train()
        loader = DataLoader(data, pin_memory=True, num_workers=4,
                            batch_size=self._batch_size)

        for x, y_output in tqdm(loader, desc="Online training",
                                total=len(loader)):
            if self._gpus:
                x = x.cuda()
                y_output = y_output.cuda()

            self._online_optimizer.zero_grad()

            y_hat = self._substitute_model(x)
            loss = self._train_loss(y_hat, y_output)
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
                      ((y_output.sum(dim=0) / self._k) - self._y_avg)

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
        if self._reward_type == "cert":
            return self._reward_cert(y_output)
        elif self._reward_type == "div":
            return self._reward_div(y_output, iteration)
        elif self._reward_type == "loss":
            return self._reward_loss(y_output, y_hat)
        else:
            return self._reward_all(y_output, y_hat, iteration)

    def _adaptive_strategy(self):
        # Number of actions
        if isinstance(self._sub_dataset, Subset):
            self._num_actions = len(
                    np.unique(self._sub_dataset.dataset.targets))
        else:
            self._num_actions = len(
                    np.unique(self._sub_dataset.targets))

        # We need to keep an average version of the victim output
        if self._reward_type == "div" or self._reward_type == "all":
            self._y_avg = torch.zeros(self._num_classes)

        # We need to keep an average and variance version of rewards
        if self._reward_type == "all":
            self._reward_avg = np.zeros(3)
            self._reward_var = np.zeros(3)

        # Implement the bandit gradients algorithm
        h_func = torch.zeros(self._num_actions)
        learning_rate = torch.zeros(self._num_actions)
        probs = np.ones(self._num_actions) / self._num_actions
        query_sets = []

        avg_reward = np.array(0.0)
        for it in range(1, self._iterations + 1):
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
                                             self._output_type)

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
                "KnockOff's attack budget: {}".format(self._budget))

        if self._sampling_strategy == "random":
            self._logger.info("Starting random sampling strategy")
            transfer_data = self._random_strategy()
        else:
            original_state_dict = self._substitute_model.state_dict()
            self._logger.info("Starting adaptive sampling strategy with {} "
                              "reward type".format(self._reward_type))
            transfer_data = self._adaptive_strategy()
            self._substitute_model.load_state_dict(original_state_dict)

        self._logger.info("Offline training of substitute model")
        self._train_model(self._substitute_model, self._optimizer,
                          transfer_data)

        self._logger.info("Test set metrics")
        vict_test_acc, vict_test_loss = self._test_model(self._victim_model,
                                                         self._test_set)
        sub_test_acc, sub_test_loss = self._test_model(self._substitute_model,
                                                       self._test_set)
        self._logger.info(
                "Victim model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        vict_test_acc, vict_test_loss))
        self._logger.info(
                "Substitute model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        sub_test_acc, sub_test_loss))

        # with open(self._save_loc + "/selected_idxs.pkl", 'wb') as f:
        #     pickle.dump(self._selected_idxs, f)
        #
        # with open(self._save_loc + "/selected_action.pkl", 'wb') as f:
        #     pickle.dump(self._selected_action, f)

        self._get_aggreement_score()

        self._save_final_subsitute()

        return
