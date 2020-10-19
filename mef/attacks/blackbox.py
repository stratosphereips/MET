import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import CustomDataset, NoYDataset


class BlackBox(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 iterations=6, lmbda=0.1, training_epochs=10,
                 batch_size=64, save_loc="./cache/blackbox"):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        train_loss = torch.nn.CrossEntropyLoss()
        test_loss = train_loss

        super().__init__(victim_model, substitute_model, optimizer,
                         train_loss, test_loss,
                         training_epochs=training_epochs,
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
            list_derivatives.append(derivative.cpu().squeeze(dim=0))

        return list_derivatives

    def _jacobian_augmentation(self, query_sets, lmbda):
        sub_dataset = ConcatDataset(query_sets)
        loader = DataLoader(sub_dataset, pin_memory=True, num_workers=4,
                            batch_size=self._batch_size)
        x_query_set = []
        for x_sub, y_sub in tqdm(loader, desc="Jacobian augmentation",
                                 total=len(loader)):
            grads = self._jacobian(x_sub)

            for idx in range(grads[0].shape[0]):
                # Select gradient corresponding to the label predicted by the
                # oracle
                grad = grads[y_sub[idx]][idx]

                # Compute sign matrix
                grad_val = torch.sign(grad)

                # Create new synthetic point in adversary substitute
                # training set
                x_new = x_sub[idx][0] + lmbda * grad_val
                x_query_set.append(x_new.detach())

        return torch.stack(x_query_set)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info("########### Starting BlackBox attack ###########")
        # Get attack's budget
        budget = len(self._sub_dataset) * (2 ** self._iterations) - \
                 len(self._sub_dataset)
        self._logger.info("BlackBox's attack budget: {}".format(budget))

        query_sets = [self._sub_dataset]
        for it in range(self._iterations):
            self._logger.info("---------- Iteration: {} ----------".format(
                    it + 1))

            train_set = ConcatDataset(query_sets)
            self._train_model(self._substitute_model, self._optimizer,
                              train_set, iteration=it)

            if it < self._iterations - 1:
                self._substitute_model.eval()
                self._logger.info("Augmenting training data")
                x_query_set = self._jacobian_augmentation(query_sets,
                                                          self._lmbda)

                self._logger.info("Labeling substitute training data")
                y_query_set = self._get_predictions(self._victim_model,
                                                    NoYDataset(x_query_set))
                # Adversary has access only to labels
                y_query_set = torch.argmax(y_query_set, dim=1)
                query_sets.append(CustomDataset(x_query_set, y_query_set))

        self._get_aggreement_score()

        self._save_final_subsitute()

        return
