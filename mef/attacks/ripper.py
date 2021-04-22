import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterator, Tuple, Union, Optional

import numpy as np
import torch
from mef.attacks.base import AttackBase
from mef.utils.pytorch.lighting.module import Generator, TrainableModel, VictimModel
from mef.utils.settings import AttackSettings
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm

from mef.utils.pytorch.datasets import SavedDataset


# TODO: rework these both random and optimized datasets so they don't have to use __len__
class _GeneratorRandomDataset(IterableDataset):
    def __init__(
        self,
        generator: Generator,
        victim_model: VictimModel,
        batch_size: int,
        batches_per_epoch: int,
    ):
        self._generator = generator
        self._victim_model = victim_model
        self._batch_size = batch_size
        self._batches_per_epoch = batches_per_epoch

    def __len__(self) -> int:
        return self._batches_per_epoch

    def _get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self._batches_per_epoch):
            u = 3
            latent_vectors = np.random.uniform(
                -u, u, size=(self._batch_size, self._generator.latent_dim)
            )
            latent_vectors = torch.from_numpy(latent_vectors.astype(np.float32))
            images = self._generator(latent_vectors)

            with torch.no_grad():
                labels = self._victim_model(images)[0]

            yield images, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Generates batch_size * batches_per_epoch labels each training epoch
        return iter(self._get_sample())


class _GeneratorOptimizedDataset(IterableDataset):
    def __init__(
        self,
        generator: Generator,
        victim_model: VictimModel,
        batch_size: int,
        batches_per_epoch: int,
        population_size: int,
        max_iterations: int,
        threshold_type: str,
        threshold_value: float,
    ):
        self._generator = generator
        self._victim_model = victim_model
        self._batch_size = batch_size
        self._batches_per_epoch = batches_per_epoch

        self._population_size = population_size
        # Parameters for optimization loop
        self._max_iterations = max_iterations
        self._threshold_type = threshold_type
        self._threshold_value = threshold_value

    @staticmethod
    def _loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y_hat - y).pow(2).sum()

    def _optimization_condition(self, c: float, it: int) -> bool:
        if self._threshold_type == "loss":
            # Loss threshold
            return c >= self._threshold_value and it < self._max_iterations
        else:
            # Confidence threshold
            return c < self._threshold_value and it < self._max_iterations

    def __len__(self) -> int:
        return self._batches_per_epoch

    def _optimize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = []
        labels = []

        # Hyperparameters from paper
        u = 3
        for _ in range(self._batch_size):
            c = np.inf if self._threshold_type == "loss" else 0
            it = 0
            specimens = np.random.uniform(
                -u, u, size=(self._population_size, self._generator.latent_dim)
            )
            specimens = specimens.astype(np.float32)
            target_label = np.random.randint(
                self._victim_model.num_classes, size=(1, 1)
            )
            y = torch.eye(self._victim_model.num_classes)[target_label]
            while self._optimization_condition(c, it):
                it += 1
                with torch.no_grad():
                    images = self._generator(torch.from_numpy(specimens))

                    y_hats = self._victim_model(images)[0]
                    y_hats = y_hats.detach().cpu()

                losses = torch.stack([self._loss(y_hat, y) for y_hat in y_hats])
                sorted_losses, sorted_idx = torch.sort(losses)
                # Select min
                image = images[sorted_idx[0]]
                label = y_hats[sorted_idx[0]]

                # select k (elite size) fittest specimens
                specimens = specimens[sorted_idx[:10]]
                specimens = np.concatenate(
                    [
                        specimens,
                        specimens
                        + np.random.normal(
                            scale=1, size=(10, self._generator.latent_dim)
                        ).astype(np.float32),
                        specimens
                        + np.random.normal(
                            scale=1, size=(10, self._generator.latent_dim)
                        ).astype(np.float32),
                    ]
                )
                c = (
                    sorted_losses[0]
                    if self._threshold_type == "loss"
                    else label[target_label]
                )

            batch.append(image)
            labels.append(label)

        return torch.stack(batch), torch.stack(labels)

    def _get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self._batches_per_epoch):
            images, labels = self._optimize()
            yield images, labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Generates batch_size * 100 labels each training epoch
        return iter(self._get_sample())


@dataclass
class RipperSettings(AttackSettings):
    latent_dim: int
    generated_data: str
    batches_per_epoch: int
    population_size: int
    # Parameters for optimization loop
    max_iterations: int
    threshold_type: str
    threshold_value: float
    save_dataset: bool
    dataset_save_loc: Path

    def __init__(
        self,
        generated_data: str,
        batches_per_epoch: int,
        population_size: int,
        max_iterations: int,
        threshold_type: str,
        threshold_value: float,
        save_dataset: bool,
        dataset_save_loc: str,
    ):
        self.generated_data = generated_data
        self.batches_per_epoch = batches_per_epoch
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.threshold_type = threshold_type.lower()
        self.threshold_value = threshold_value
        self.save_dataset = save_dataset
        self.dataset_save_loc = Path(dataset_save_loc)

        # Check configuration
        if self.generated_data not in ["random", "optimized"]:
            raise ValueError(
                "Ripper's generated_data must be one of {random, optimized}"
            )

        if self.threshold_type not in ["loss", "confidence"]:
            raise ValueError(
                "Ripper's threshold_type must be one of {loss, confidence}"
            )


class Ripper(AttackBase):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        generator: Generator,
        generated_data: str = "optimized",
        batches_per_epoch: int = 100,
        population_size: int = 30,
        max_iterations: int = 10,
        threshold_type: str = "loss",
        threshold_value: float = 0.02,
        save_dataset: bool = False,
        dataset_save_loc: str = "./cache",
        *args: Union[int, bool, Path],
        **kwargs: Union[int, bool, Path],
    ):
        super().__init__(victim_model, substitute_model, *args, **kwargs)
        self.attack_settings = RipperSettings(
            generated_data,
            batches_per_epoch,
            population_size,
            max_iterations,
            threshold_type,
            threshold_value,
            save_dataset,
            dataset_save_loc,
        )
        # Ripper's specific attributes
        self._generator = generator
        self._val_set = None

    @classmethod
    def _get_attack_parser(cls):
        parser = argparse.ArgumentParser(description="Ripper attack")
        parser.add_argument(
            "--generated_data",
            default="optimized",
            type=str,
            help="Type of generated data from generator. Can be one of {random, optimized} (Default: optimized)",
        )
        parser.add_argument(
            "--batches_per_epoch",
            default=100,
            type=int,
            help="How many batches should be created per epochs. (Default: 100)",
        )
        parser.add_argument(
            "--population_size",
            default=30,
            type=int,
            help="Population size to be used in the evolutionary algorithm. (Default: 10)",
        )
        parser.add_argument(
            "--max_iterations",
            default=10,
            type=int,
            help="Maximum number of iteration of the evolutionary algorithm. (Default: 10)",
        )
        # TODO: add better explanation
        parser.add_argument(
            "--threshold_type",
            default="loss",
            type=str,
            help="Type of threhold for the evolutionary algorithm. Can be one of {loss, confidence} (Default: loss)",
        )
        parser.add_argument(
            "--threshold_value",
            default=0.02,
            type=float,
            help="Value for threshold. (Default: 0.02)",
        )
        parser.add_argument(
            "--save_dataset",
            action="store_true",
            help="Instead of constantly creating new samples, create the samples once, save them and resuse them. The number of saved samples is batch_size * batches_per_epoch. (Default: False)",
        )
        parser.add_argument(
            "--dataset_save_loc",
            default="./cache/",
            type=str,
            help="Name of location, where the dataset samples should be saved. (Default: ./cache/)",
        )

        return parser

    def _create_dataset(self) -> Dataset:
        labels_filepath = self.attack_settings.dataset_save_loc.joinpath("labels.pl")
        if not self.attack_settings.dataset_save_loc.joinpath("complete").exists():
            self.attack_settings.dataset_save_loc.mkdir(parents=True, exist_ok=True)
            dataset_loader = DataLoader(dataset=self._thief_dataset)
            all_labels = []
            i = 0
            for images, labels in tqdm(
                dataset_loader, desc="Creating synthetic dataset"
            ):
                # Dataloader adds one dimension
                images = images.squeeze(dim=0)
                labels = labels.squeeze(dim=0)

                all_labels.append(labels)
                for image in images:
                    # Saving as tensor since torchvision image since torchvision save image is not working nicely with normalized images
                    torch.save(
                        image.detach().cpu(),
                        self.attack_settings.dataset_save_loc.joinpath(f"{i}.pt"),
                    )
                    i += 1

            all_labels = torch.cat(all_labels).detach().cpu()
            pickle.dump(all_labels, open(labels_filepath, "wb"))

            # Create empty file as flag that the dataset is already created
            self.attack_settings.dataset_save_loc.joinpath("complete").touch()

        labels = pickle.load(open(labels_filepath, "rb"))

        return SavedDataset(self.attack_settings.dataset_save_loc, labels)

    def _get_student_dataset(self) -> IterableDataset:
        self._generator.eval()
        self._generator._generator.eval()
        self._victim_model.eval()
        self._victim_model.model.eval()
        if self.attack_settings.generated_data == "random":
            return _GeneratorRandomDataset(
                self._generator,
                self._victim_model,
                self.base_settings.batch_size,
                self.attack_settings.batches_per_epoch,
            )
        else:
            return _GeneratorOptimizedDataset(
                self._generator,
                self._victim_model,
                self.base_settings.batch_size,
                self.attack_settings.batches_per_epoch,
                self.attack_settings.population_size,
                self.attack_settings.max_iterations,
                self.attack_settings.threshold_type,
                self.attack_settings.threshold_value,
            )

    def _check_args(self, test_set: Dataset, val_set: Optional[Dataset] = None) -> None:
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        if val_set is not None:
            if not isinstance(val_set, Dataset):
                self._logger.error("Test set must be Pytorch's dataset.")
                raise TypeError()
            self._val_set = val_set

        self._test_set = test_set

        return

    def _run(self, test_set: Dataset, val_set: Optional[Dataset] = None) -> None:
        self._check_args(test_set, val_set)
        self._logger.info("########### Starting Ripper attack ##########")
        # Get budget of the attack
        self._logger.info(
            "Ripper's attack budget: {}".format(
                self.trainer_settings.training_epochs
                * self.base_settings.batch_size
                * self.attack_settings.batches_per_epoch
            )
        )

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        if self.attack_settings.save_dataset:
            self._thief_dataset = self._create_dataset()

        # Random and optimized datasets rise warnings with number of workes and __len__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._train_substitute_model(self._thief_dataset, self._val_set)

        return
