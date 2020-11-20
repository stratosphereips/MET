import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from mef.attacks.base import Base
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.settings import AttackSettings


def loss(y_hat,
         y):
    return np.sum(np.power(y_hat - y, 2))


def optimize(classifier,
             generator,
             batch_size,
             latent_dim,
             num_classes):
    batch = []

    # Hyperparameters from paper
    u = 3
    t = 0.02  # t - threshold value
    pop_size = 30  # K - population size
    max_iterations = 10
    for _ in range(batch_size):
        c = np.inf
        it = 0
        image = None
        specimens = np.random.uniform(-u, u, size=(pop_size, latent_dim))
        label = np.random.randint(num_classes, size=(1, 1))
        y = np.eye(num_classes)[label]
        while c >= t and it < max_iterations:
            it += 1
            with torch.no_grad():
                images = generator(torch.tensor(specimens).float())

                # The original implementation expects the classifier to
                # return logits
                logits = classifier(images).cpu()
                y_hats = F.softmax(logits, dim=-1).numpy()

            losses = [loss(y_hat, y) for y_hat in y_hats]
            indexes = np.argsort(losses)
            image = images[indexes[0]]
            # select k (elite size) fittest specimens
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=1, size=(10, latent_dim)),
                specimens + np.random.normal(scale=1, size=(10, latent_dim))])
            c = np.amin(losses)

        batch.append(image)

    return torch.stack(batch)


class GeneratorRandomDataset(IterableDataset):
    def __init__(self,
                 generator,
                 latent_dim,
                 victim_model,
                 batch_size,
                 output_type="softmax",
                 to_greyscale=False):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        self._output_type = output_type
        # self._to_greyscale = to_greyscale
        self._batch_size = batch_size

    def __iter__(self):
        for _ in range(1000):
            images = self._generator(torch.Tensor(
                    np.random.uniform(-3.3, 3.3, size=(self._batch_size,
                                                       self._latent_dim))))

            # if self._to_greyscale:
            #     multipliers = [.2126, .7152, .0722]
            #     multipliers = np.expand_dims(multipliers, 0)
            #     multipliers = np.expand_dims(multipliers, -1)
            #     multipliers = np.expand_dims(multipliers, -1)
            #     multipliers = np.tile(multipliers, [1, 1, 32, 32])
            #     multipliers = torch.Tensor(multipliers)
            #     images = images * multipliers
            #     images = images.sum(axis=1, keepdims=True)

            with torch.no_grad():
                y_preds = self._victim_model(images)
                if self._output_type == "one_hot":
                    labels = F.one_hot(torch.argmax(y_preds, dim=-1),
                                       num_classes=y_preds.size()[1])
                    # to_oneshot returns tensor with uint8 type
                    labels = labels.float()
                elif self._output_type == "softmax":
                    labels = F.softmax(y_preds, dim=-1)
                elif self._output_type == "labels":
                    labels = torch.argmax(y_preds, dim=-1)
                else:
                    labels = y_preds

            yield images, labels


class GeneratorOptimizedDataset(IterableDataset):
    def __init__(
            self,
            generator,
            latent_dim,
            victim_model,
            batch_size,
            num_classes,
            output_type="softmax",
            to_grayscale=False):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        self._output_type = output_type
        # self._to_grayscale = to_grayscale
        self._batch_size = batch_size
        self._num_classes = num_classes

    def __iter__(self):
        # optimization = optimize_to_grayscale if self._to_grayscale else \
        #     optimize
        optimization = optimize
        for _ in range(100):
            images = optimization(self._victim_model, self._generator,
                                  self._batch_size, self._latent_dim,
                                  self._num_classes)

            with torch.no_grad():
                y_preds = self._victim_model(images)
                if self._output_type == "one_hot":
                    labels = F.one_hot(torch.argmax(y_preds, dim=-1),
                                       num_classes=y_preds.size()[1])
                    # to_oneshot returns tensor with uint8 type
                    labels = labels.float()
                elif self._output_type == "softmax":
                    labels = F.softmax(y_preds, dim=-1)
                elif self._output_type == "labels":
                    labels = torch.argmax(y_preds, dim=-1)
                else:
                    labels = y_preds

            yield images, labels


@dataclass
class RipperSettings(AttackSettings):
    latent_dim: int
    generated_data: str
    output_type: str
    budget: int

    def __init__(self,
                 latent_dim: int,
                 generated_data: str,
                 output_type: str,
                 budget: int):
        self.latent_dim = latent_dim
        self.generated_data = generated_data
        self.output_type = output_type.lower()
        self.budget = budget

        # Check configuration
        if self.generated_data not in ["random", "optimized"]:
            raise ValueError("Ripper's generated_data must be one of {random, "
                             "optimized}")


class Ripper(Base):
    def __init__(self,
                 victim_model,
                 substitute_model,
                 generator,
                 latent_dim,
                 num_classes,
                 generated_data="optimized",
                 output_type="softmax",
                 budget=20000):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        loss = soft_cross_entropy

        attack_settings = RipperSettings(latent_dim, generated_data,
                                         output_type, budget)
        super().__init__(victim_model, substitute_model, optimizer, loss,
                         num_classes)
        self.attack_settings = attack_settings

        # Ripper's specific attributes
        self._generator = generator

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="Ripper attack")
        parser.add_argument("--generated_data", default="optimized", type=str,
                            help="Type of generated data from generator. Can "
                                 "be one of {random, optimized} (Default: "
                                 "optimized)")
        parser.add_argument("--output_type", default="softmax", type=str,
                            help="Type of output from victim model {softmax, "
                                 "logits, one_hot, labels} (Default: softmax)")
        parser.add_argument("--training_epochs", default=200, type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 200)")

        cls._add_base_args(parser)

        return parser

    def _get_student_dataset(self):
        self._generator.eval()
        self._victim_model.eval()
        if self.attack_settings.generated_data == "random":
            return GeneratorRandomDataset(self._generator,
                                          self.attack_settings.latent_dim,
                                          self._victim_model,
                                          self.data_settings.batch_size,
                                          self.attack_settings.output_type)
        else:
            return GeneratorOptimizedDataset(self._generator,
                                             self.attack_settings.latent_dim,
                                             self._victim_model,
                                             self.data_settings.batch_size,
                                             self._num_classes,
                                             self.attack_settings.output_type)

    def _run(self, *args, **kwargs):
        self._logger.info("########### Starting Ripper attack ##########")
        # Get budget of the attack
        self._logger.info("Ripper's attack budget: {}".format(
                self.attack_settings.budget))

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        self._train_substitute_model(self._thief_dataset, self._test_set)

        return
