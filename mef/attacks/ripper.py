import argparse
from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from mef.attacks.base import Base
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.settings import AttackSettings


def loss(y_hat,
         y):
    return np.sum(np.power(y_hat - y, 2))


def optimize(victim_model,
             generator,
             batch_size,
             latent_dim,
             num_classes):
    batch = []
    labels = []

    # Hyperparameters from paper
    u = 3
    t = 0.02  # t - threshold value
    pop_size = 30  # K - population size
    max_iterations = 10
    for _ in range(batch_size):
        c = np.inf
        it = 0
        image = None
        specimens = np.random.uniform(-u, u,
                                      size=(pop_size, latent_dim)
                                      ).astype(np.float32)
        target_label = np.random.randint(num_classes, size=(1, 1))
        y = np.eye(num_classes)[target_label].astype(np.float32)
        while c >= t and it < max_iterations:
            it += 1
            with torch.no_grad():
                images = generator(torch.from_numpy(specimens))

                # The original implementation expects the classifier to
                # return logits
                y_hats = victim_model(images)[0].detach().cpu().numpy()

            losses = [loss(y_hat, y) for y_hat in y_hats]
            indexes = np.argsort(losses)
            image = images[indexes[0]]
            label = y_hats[indexes[0]]
            # select k (elite size) fittest specimens
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=1,
                                             size=(10, latent_dim)
                                             ).astype(np.float32),
                specimens + np.random.normal(scale=1,
                                             size=(10, latent_dim)
                                             ).astype(np.float32)])
            c = np.amin(losses)

        batch.append(image)
        labels.append(torch.from_numpy(label))

    return torch.stack(batch), torch.stack(labels)


class GeneratorRandomDataset(IterableDataset):
    def __init__(self,
                 generator,
                 latent_dim,
                 victim_model,
                 batch_size):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        self._batch_size = batch_size

    def __iter__(self):
        u = 3
        for _ in range(100):
            latent_vectors = np.random.uniform(-u, u, size=(self._batch_size,
                                                            self._latent_dim))
            latent_vectors = torch.from_numpy(latent_vectors)
            images = self._generator(latent_vectors)

            with torch.no_grad():
                labels = self._victim_model(images)

            yield images, labels


class GeneratorOptimizedDataset(IterableDataset):
    def __init__(
            self,
            generator,
            latent_dim,
            victim_model,
            batch_size,
            num_classes):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        self._batch_size = batch_size
        self._num_classes = num_classes

    def __iter__(self):
        for _ in range(100):
            images, labels = optimize(self._victim_model, self._generator,
                                      self._batch_size, self._latent_dim,
                                      self._num_classes)
            yield images, labels


@dataclass
class RipperSettings(AttackSettings):
    latent_dim: int
    generated_data: str

    def __init__(self,
                 latent_dim: int,
                 generated_data: str):
        self.latent_dim = latent_dim
        self.generated_data = generated_data

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
                 generated_data="optimized"):

        super().__init__(victim_model, substitute_model)
        self.attack_settings = RipperSettings(latent_dim, generated_data)

        # Ripper's specific attributes
        self._generator = generator

    @classmethod
    def _get_attack_parser(cls):
        parser = argparse.ArgumentParser(description="Ripper attack")
        parser.add_argument("--generated_data", default="optimized", type=str,
                            help="Type of generated data from generator. Can "
                                 "be one of {random, optimized} (Default: "
                                 "optimized)")

        return parser

    def _get_student_dataset(self):
        self._generator.eval()
        self._victim_model.eval()
        if self.attack_settings.generated_data == "random":
            return GeneratorRandomDataset(self._generator,
                                          self.attack_settings.latent_dim,
                                          self._victim_model,
                                          self.base_settings.batch_size)
        else:
            return GeneratorOptimizedDataset(self._generator,
                                             self.attack_settings.latent_dim,
                                             self._victim_model,
                                             self.base_settings.batch_size,
                                             self._victim_model.num_classes)

    def _check_args(self,
                    test_set: Type[Dataset]):
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._test_set = test_set

        return

    def _run(self,
             test_set: Type[Dataset]):
        self._check_args(test_set)
        self._logger.info("########### Starting Ripper attack ##########")
        # Get budget of the attack
        self._logger.info("Ripper's attack budget: {}".format(
                self.trainer_settings.training_epochs *
                self.base_settings.batch_size * 100))

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        self._train_substitute_model(self._thief_dataset, self._test_set)

        return
