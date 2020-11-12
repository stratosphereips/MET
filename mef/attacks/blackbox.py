from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from mef.attacks.base import AttackSettings, Base
from mef.utils.pytorch.datasets import CustomDataset, NoYDataset


@dataclass
class BlackBoxSettings(AttackSettings):
    iterations: int
    lmbda: float

    def __init__(self,
                 iterations: int,
                 lmbda: float):
        self.iterations = iterations
        self.lmbda = lmbda


class BlackBox(Base):

    def __init__(self,
                 victim_model,
                 substitute_model,
                 num_classes,
                 iterations=6,
                 lmbda=0.1):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, optimizer,
                         loss)
        self.attack_settings = BlackBoxSettings(iterations, lmbda)
        self.trainer_settings.validation = False
        self.data_settings._num_classes = num_classes

    def _jacobian(self, x):
        list_derivatives = []
        x_var = x.requires_grad_()

        predictions = self._substitute_model(x_var)
        for class_idx in range(self.data_settings._num_classes):
            outputs = predictions[:, class_idx]
            derivative = torch.autograd.grad(
                    outputs,
                    x_var,
                    grad_outputs=torch.ones_like(outputs),
                    retain_graph=True)[0]
            list_derivatives.append(derivative.cpu().squeeze(dim=0))

        return list_derivatives

    def _jacobian_augmentation(self, query_sets, lmbda):
        thief_dataset = ConcatDataset(query_sets)
        loader = DataLoader(thief_dataset, pin_memory=True, num_workers=4,
                            batch_size=self.data_settings.batch_size)
        x_query_set = []
        for x_thief, y_thief in tqdm(loader, desc="Jacobian augmentation",
                                     total=len(loader)):
            grads = self._jacobian(x_thief)

            for idx in range(grads[0].shape[0]):
                # Select gradient corresponding to the label predicted by the
                # oracle
                grad = grads[y_thief[idx]][idx]

                # Compute sign matrix
                grad_val = torch.sign(grad)

                # Create new synthetic point in adversary substitute
                # training set
                x_new = x_thief[idx][0] + lmbda * grad_val
                x_query_set.append(x_new.detach())

        return torch.stack(x_query_set)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info("########### Starting BlackBox attack ###########")
        # Get attack's budget
        budget = len(self._thief_dataset) * \
                 (2 ** self.attack_settings.iterations) - \
                 len(self._thief_dataset)
        self._logger.info("BlackBox's attack budget: {}".format(budget))

        query_sets = [self._thief_dataset]
        for it in range(self.attack_settings.iterations):
            self._logger.info("---------- Iteration: {} ----------".format(
                    it + 1))

            train_set = ConcatDataset(query_sets)
            self._train_substitute_model(train_set, iteration=it + 1)

            if it < self.attack_settings.iterations - 1:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_query_set = self._jacobian_augmentation(
                        query_sets, self.attack_settings.lmbda)

                self._logger.info("Labeling substitute training data")
                # Adversary has access only to labels
                y_query_set = self._get_predictions(self._victim_model,
                                                    NoYDataset(x_query_set),
                                                    "labels")
                query_sets.append(CustomDataset(x_query_set, y_query_set))

        self._finalize_attack()

        return
