# Based on https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob
# /main/art/attacks/extraction/knockoff_nets.py

import numpy as np
import torch
import torch.nn.functional as F

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomDataset


class KnockOff(Base):

    def __init__(self, victim_model, substitute_model, x_test, y_test,
                 num_classes, sampling_strategy="adaptive",
                 reward_type="cert", output_type="logits", budget=1000,
                 training_epochs=100, batch_size=64,
                 save_loc="./cache/knockoff"):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        train_loss = F.cross_entropy
        test_loss = train_loss

        super().__init__(victim_model, substitute_model, x_test, y_test,
                         optimizer, train_loss, test_loss, training_epochs,
                         batch_size=batch_size, num_classes=num_classes,
                         save_loc=save_loc, validation=False)

        # KnockOff's specific attributes
        self._sampling_strategy = sampling_strategy
        self._reward_type = reward_type
        self._output_type = output_type
        self._budget = budget
        self._k = 4
        self._iterations = self._budget // self._k

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

    def _random_strategy(self, x):
        self._logger.info("Selecting random sample from thief dataset of "
                          "size {}".format(self._budget))
        idx_x = np.arange(len(x))
        idx_selected = np.random.permutation(idx_x)[:self._budget]
        x_selected = x[idx_selected]

        self._logger.info("Getting fake labels from victim model")
        fake_labels = self._get_predictions(self._victim_model, x_selected)
        fake_labels = np.argmax(fake_labels, axis=1)

        return x_selected, fake_labels

    def _sample_data(self, x, y, action):
        x_ = x[y == action]
        rnd_idx = np.random.permutation(len(x_))[:self._k]

        return x_[rnd_idx]

    @staticmethod
    def _reward_cert(y_output):
        """
        Compute `cert` reward value.
        """
        largests = np.partition(y_output.flatten(), -2)[-2:]
        reward = largests[1] - largests[0]

        return reward

    def _reward_div(self, y_output: np.ndarray, n):
        """
        Compute `div` reward value.
        """
        # First update y_avg
        self._y_avg = self._y_avg + (1.0 / n) * (y_output - self._y_avg)

        # Then compute reward
        reward = 0
        for k in range(self._num_classes):
            reward += np.maximum(0, y_output[k] - self._y_avg[k])

        return reward

    def _reward_loss(self, y_output, y_hat):
        """
        Compute `loss` reward value.
        """
        # Compute victim probs
        aux_exp = np.exp(y_output)
        probs_output = aux_exp / np.sum(aux_exp)

        # Compute thieved probs
        aux_exp = np.exp(y_hat)
        probs_hat = aux_exp / np.sum(aux_exp)

        # Compute reward
        reward = 0
        for k in range(self._num_classes):
            reward += -probs_output[k] * np.log(probs_hat[k])

        return reward

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

    def _adaptive_strategy(self, x, y):
        # Number of actions
        self._num_actions = len(np.unique(y))

        # We need to keep an average version of the victim output
        if self._reward == "div" or self._reward == "all":
            self._y_avg = np.zeros(self._num_classes)

        # We need to keep an average and variance version of rewards
        if self._reward == "all":
            self._reward_avg = np.zeros(3)
            self._reward_var = np.zeros(3)

        # Implement the bandit gradients algorithm
        h_func = np.zeros(self._num_actions)
        learning_rate = np.zeros(self._num_actions)
        probs = np.ones(self._num_actions) / self._num_actions
        x_selected = []
        queried_labels = []

        avg_reward = 0.0
        for it in range(1, self._iterations + 1):
            self._logger.info("---------- Iteration: {} ----------".format(it))
            # Sample an action
            action = np.random.choice(np.arange(0, self._num_actions), p=probs)

            # Select sample to attack
            x_sampled = self._sample_data(x, y, action)
            x_selected.append(x_sampled)

            # Query the victim model
            y_output = self._get_predictions(self._victim_model, x_sampled,
                                             self._output_type)
            fake_label = np.argmax(y_output, axis=1)
            queried_labels.append(fake_label)

            # Train the thieved classifier
            train_set = CustomDataset(x_sampled, fake_label)
            self._train_model(self._substitute_model, self._optimizer,
                              train_set, training_epochs=1)

            # Test new labels
            y_hat = self._get_predictions(self._substitute_model, x_sampled,
                                          output_type="logits")

            # Compute rewards
            self._logger.info("Computing rewards")
            for y_output_, y_hat_ in zip(y_output, y_hat):
                reward = self._reward(y_output_, y_hat_, it)
                avg_reward = avg_reward + (1.0 / it) * (reward - avg_reward)

                # Update learning rate
                learning_rate[action] += 1

                # Update H function
                for a in range(self._num_actions):
                    if a != action:
                        h_func[a] = h_func[a] - 1.0 / learning_rate[action] * \
                                    (reward - avg_reward) * probs[a]
                    else:
                        h_func[a] = h_func[a] + 1.0 / learning_rate[action] * \
                                    (reward - avg_reward) * (1 - probs[a])

                # Update probs
                aux_exp = np.exp(h_func)
                probs = aux_exp / np.sum(aux_exp)

        return np.vstack(x_selected), np.hstack(queried_labels)

    def run(self, x, y):
        self._logger.info(
                "########### Starting KnockOff attack ###########")

        vict_test_acc, vict_test_loss = self._test_model(
                self._substitute_model, self._test_set)
        self._logger.info(
                "Victim model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        vict_test_acc, vict_test_loss))

        if self._sampling_strategy == "random":
            self._logger.info("Starting random sampling strategy")
            x_selected, y_selected = self._random_strategy(x)
        else:
            self._logger.info("Starting adaptive sampling strategy")
            x_selected, y_selected = self._adaptive_strategy(x, y)

        self._logger.info("Final training of substitute model")
        train_set = CustomDataset(x_selected, y_selected)
        self._train_model(self._substitute_model, self._optimizer, train_set)

        self._logger.info("Test set metrics")
        vict_test_acc, vict_test_loss = self._test_model(
                self._substitute_model, self._test_set)
        sub_test_acc, sub_test_loss = self._test_model(self._substitute_model,
                                                       self._test_set)
        self._logger.info(
                "Victim model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        vict_test_acc, vict_test_loss))
        self._logger.info(
                "Substitute model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        sub_test_acc, sub_test_loss))

        self._get_aggreement_score()

        self._save_final_subsitute()

        return
