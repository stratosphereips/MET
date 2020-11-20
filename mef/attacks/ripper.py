import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

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
        specimens = np.random.uniform(-u, u, size=(pop_size, latent_dim))
        target_label = np.random.randint(num_classes, size=(1, 1))
        y = np.eye(num_classes)[target_label]
        while c >= t and it < max_iterations:
            it += 1
            with torch.no_grad():
                images = generator(torch.from_numpy(specimens).float())

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
                specimens + np.random.normal(scale=1, size=(10, latent_dim)),
                specimens + np.random.normal(scale=1, size=(10, latent_dim))])
            c = np.amin(losses)

        batch.append(image)
        labels.append(torch.from_numpy(label))

    return torch.stack(batch), torch.stack(labels)


class GeneratorRandomDataset(IterableDataset):
    def __init__(self,
                 generator,
                 latent_dim,
                 victim_model,
                 batch_size,
                 to_greyscale=False):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        # self._to_greyscale = to_greyscale
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
            num_classes,
            to_grayscale=False):
        self._generator = generator
        self._latent_dim = latent_dim
        self._victim_model = victim_model
        # self._to_grayscale = to_grayscale
        self._batch_size = batch_size
        self._num_classes = num_classes

    def __iter__(self):
        # optimization = optimize_to_grayscale if self._to_grayscale else \
        #     optimize
        optimization = optimize
        for _ in range(100):
            images, labels = optimization(self._victim_model, self._generator,
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
                 num_classes,
                 generated_data="optimized"):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        loss = soft_cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss,
                         num_classes, victim_output_type="softmax")
        self.attack_settings = RipperSettings(latent_dim, generated_data)

        # Ripper's specific attributes
        self._generator = generator

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="Ripper attack")
        parser.add_argument("--generated_data", default="optimized", type=str,
                            help="Type of generated data from generator. Can "
                                 "be one of {random, optimized} (Default: "
                                 "optimized)")
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
                                          self.data_settings.batch_size)
        else:
            return GeneratorOptimizedDataset(self._generator,
                                             self.attack_settings.latent_dim,
                                             self._victim_model,
                                             self.data_settings.batch_size,
                                             self._num_classes)

    def _run(self, *args, **kwargs):
        self._logger.info("########### Starting Ripper attack ##########")
        # Get budget of the attack
        self._logger.info("Ripper's attack budget: {}".format(
                self.trainer_settings.training_epochs *
                self.data_settings.batch_size * 100))

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        self._train_substitute_model(self._thief_dataset, self._test_set)

        return
