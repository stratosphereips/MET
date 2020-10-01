from dataclasses import dataclass

import imgaug
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from advertorch.attacks.carlini_wagner import CarliniWagnerL2Attack
from advertorch.attacks.iterative_projected_gradient import PGDAttack
from imgaug import augmenters as iaa
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from .base import Base
from ..utils.config import Configuration
from ..utils.pytorch.datasets import AugmentationDataset, CustomDataset


# Function so each worker generates different augmented data
def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


@dataclass
class ActiveThiefConfig:
    adversial_strategy: str = "pgd"
    set_size: int = 200
    augmentation_multiplier: int = 1
    iterations: int = 5


class CloudLeak(Base):

    def __init__(self, victim_model, substitute_model, test_set, thief_dataset,
                 num_classes, save_loc="./cache/cloudleak"):
        super().__init__(save_loc)

        # Get ActiveThief's configuration
        self._config = Configuration.get_configuration(ActiveThiefConfig,
                                                       "attacks/cloudleak")
        self._adversial_strategy = self._config.adversial_strategy.lower()

        # Datasets
        self._test_set = test_set
        self._thief_dataset = thief_dataset

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        if self._test_config.gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        # Dataset information
        self._num_classes = num_classes
        self._budget = self._config.iterations * self._config.set_size

        # Optimizer, loss_functions
        self._optimizer = torch.optim.Adam(self._substitute_model.parameters())
        self._train_loss = F.cross_entropy
        self._test_loss = F.cross_entropy

        # Check configuration
        if self._adversial_strategy not in ["random", "pgd", "cw", "fa", "ff"]:
            self._logger.error(
                    "CloudLeak's adversial strategy must be one of {random, "
                    "pgd, cw, fa, ff}")
            raise ValueError()

    def _craft_adversial_samples(self, samples):
        if self._adversial_strategy == "pgd":
            attack = PGDAttack(self._substitute_model)
        elif self._adversial_strategy == "cw":
            attack = CarliniWagnerL2Attack(self._substitute_model,
                                           self._num_classes)
        else:
            self._logger.error(
                    "CloudLeak's adversial strategy must be one of {random, "
                    "pgd, cw, fa, ff}")
            raise ValueError()

        loader = DataLoader(samples,
                            batch_size=self._test_config.batch_size // 2,
                            pin_memory=True,
                            num_workers=4)

        adversary_samples = []
        for _, batch in enumerate(
                tqdm(loader, desc="Getting adversial samples")):
            x, _ = batch
            adversary_samples.append(attack.perturb(x).cpu())

        return torch.cat(adversary_samples)

    def run(self):
        self._logger.info("########### Starting CloudLeak attack ###########")
        # Get budget of the attack
        self._logger.info("CloudLeak's attack budget: {}".format(self._budget))

        # Get victim model predicted labels for test dataset
        self._logger.info("Getting victim model's labels for test set")
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=1).numpy()

        self._logger.info(
                "Number of test samples: {}".format(len(vict_test_labels)))

        # Get victim model test dataset metrics
        self._logger.info("Getting victim model's metrics for test set")
        vict_test_acc, vict_test_loss = self._test_model(self._victim_model,
                                                         self._test_loss,
                                                         self._test_set)

        available_samples = set(range(len(self._thief_dataset)))
        iteration_sets = []
        for iteration in range(self._config.iterations):
            self._logger.info(
                    "---------- Iteration: {} ----------".format(
                            iteration + 1))

            # Get metrics from victim model and substitute model
            self._logger.info(
                    "Getting substitute model's metrics for test set")
            sub_test_acc, sub_test_loss = self._test_model(
                    self._substitute_model, self._test_loss, self._test_set)
            self._logger.info("Test set metrics")
            self._logger.info(
                    "Victim model Accuracy: {:.1f}% Loss: {:.3f}".format(
                            vict_test_acc, vict_test_loss))
            self._logger.info(
                    "Substitute model Accuracy: {:.1f}% Loss: {:.3f}".format(
                            sub_test_acc, sub_test_loss))

            # Step 1: Randomly select images for current iteration
            self._logger.info("Preparing initial random query set")
            idx = np.random.choice(np.arange(len(available_samples)),
                                   size=self._config.set_size, replace=False)
            available_samples -= set(idx)
            iteration_sets.append(Subset(self._thief_dataset, idx))

            # Step 2: Craft adversial examples
            self._logger.info(
                    "Crafting adversial samples with {} method on substitute "
                    "model".format(self._adversial_strategy))
            adversial_samples = self._craft_adversial_samples(
                    iteration_sets[iteration])

            # Step 3: Generate synthetic dataset
            self._logger.info("Generating synthetic dataset with adversial "
                              "samples and victim model")
            dummy_labels = torch.ones(len(adversial_samples))
            unlabeled_synth = CustomDataset(adversial_samples, dummy_labels)
            synth_labels = self._get_predictions(self._victim_model,
                                                 unlabeled_synth)
            synth_labels = torch.argmax(synth_labels, dim=1)

            # Split current set into train and validation
            if isinstance(self._test_config.val_set_size, float):
                train_set_size = int(self._config.set_size *
                                     (1 - self._test_config.val_set_size))
            elif isinstance(self._test_config.val_set_size, int):
                train_set_size = int(self._config.set_size -
                                     self._test_config.val_set_size)
            else:
                raise ValueError("split_size must be either float or integer!")

            indices = torch.randperm(self._config.set_size).tolist()
            train_set_data = adversial_samples[indices[:train_set_size]]
            train_set_labels = synth_labels[indices[:train_set_size]]
            val_set_data = adversial_samples[indices[train_set_size:]]
            val_set_labels = synth_labels[indices[train_set_size:]]

            # Add data augmentation
            transform = transforms.Compose([iaa.Sequential(
                    [iaa.Fliplr(p=0.5),
                     iaa.Flipud(p=0.5),
                     iaa.Affine(rotate=(-45, 45)),
                     iaa.ScaleX((0.5, 1.5)),
                     iaa.ScaleY((0.5, 1.5))
                     ]).augment_image, transforms.ToTensor()])

            train_set_list = [CustomDataset(train_set_data, train_set_labels)]
            val_set_list = [CustomDataset(val_set_data, val_set_labels)]
            for _ in range(self._config.augmentation_multiplier):
                train_set_list.append(
                        AugmentationDataset(train_set_data, train_set_labels,
                                            transform))
                val_set_list.append(
                        AugmentationDataset(val_set_data, val_set_labels,
                                            transform))

            train_set = ConcatDataset(train_set_list)
            val_set = ConcatDataset(val_set_list)

            # Step 4: Train substitute model
            self._logger.info(
                    "Training substitute model with synthetic dataset")
            self._train_model(self._substitute_model, self._optimizer,
                              self._train_loss, train_set, val_set,
                              iteration + 1, worker_init_fn)

            # TODO: change to ATE metric
            # Agreement score
            self._logger.info("Getting attack metric")
            self._get_attack_metric(self._substitute_model, self._test_set,
                                    vict_test_labels)
