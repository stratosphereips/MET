import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomDataset, NoYDataset


class BlackBox(Base):

    def __init__(self, victim_model, substitute_model, x_test, y_test,
                 num_classes, iterations=6, lmbda=0.1, training_epochs=10,
                 batch_size=64, save_loc="./cache/blackbox"):
        optimizer = torch.optim.Adam(self._substitute_model.parameters())
        train_loss = F.cross_entropy
        test_loss = train_loss

        super().__init__(victim_model, substitute_model, x_test, y_test,
                         optimizer, train_loss, test_loss, training_epochs,
                         batch_size=batch_size, save_loc=save_loc,
                         num_classes=num_classes, validation=False)

        # BlackBox's specific attributes
        self._iterations = iterations
        self._lmbda = lmbda

    def _jacobian(self, x):
        list_derivatives = []
        x_var = x.requires_grad_()

        predictions = self._substitute_model(x_var)
        for class_idx in range(self._num_classes):
            outputs = predictions[:, class_idx]
            derivative = torch.autograd.grad(
                    outputs,
                    x_var,
                    grad_outputs=torch.ones_like(outputs),
                    retain_graph=True)[0]
            list_derivatives.append(derivative.cpu().squeeze(dim=0).numpy())

        return list_derivatives

    def _jacobian_augmentation(self, x_old, y_sub, lmbda):
        x_new = np.vstack([x_old, x_old])
        data = NoYDataset(x_new)
        loader = DataLoader(data, pin_memory=True, num_workers=4,
                            batch_size=self._batch_size)

        for _, x in tqdm(enumerate(loader), desc="Jacobian augmentation",
                         total=len(loader)):

            grads = self._jacobian(x)

            for idx in range(grads[0].shape[0]):
                # Select gradient corresponding to the label predicted by the
                # oracle
                grad = grads[y_sub[idx]][idx]

                # Compute sign matrix
                grad_val = np.sign(grad)

                # Create new synthetic point in adversary substitute
                # training set
                x_new[len(x_old) + idx] = x_new[idx] + lmbda * grad_val

        return x_new

    def run(self, x, y):
        self._logger.info("########### Starting BlackBox attack ###########")
        # Get attack's budget
        budget = len(x) * (2 ** self._iterations) - len(x)
        self._logger.info("BlackBox's attack budget: {}".format(budget))

        x_sub = x
        y_sub = y
        for it in range(self._iterations):
            self._logger.info("---------- Iteration: {} ----------".format(
                    it + 1))

            train_set = CustomDataset(x_sub, y_sub)
            self._train_model(self._substitute_model, self._optimizer,
                              self._train_loss, train_set, iteration=it)

            if it < self._iterations - 1:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_sub = self._jacobian_augmentation(x_sub, y_sub, self._lmbda)

                self._logger.info("Labeling substitute training data")
                x_sub_new = x_sub[len(x_sub) // 2:]
                y_sub_new = self._get_predictions(self._victim_model,
                                                  x_sub_new)
                # Adversary has access only to labels
                y_sub_new = np.argmax(y_sub_new, axis=1)
                y_sub = np.hstack([y_sub, y_sub_new])

        self._get_aggreement_score()

        self._save_final_subsitute()

        return
