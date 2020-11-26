import argparse

import torch
import torch.nn.functional as F

from mef.utils.pytorch.datasets import CustomLabelDataset
from .base import Base


class CopyCat(Base):

    def __init__(self,
                 victim_model,
                 substitute_model,
                 num_classes,
                 optimizer: torch.optim.Optimizer = None,
                 loss=None,
                 lr_scheduler=None):
        if optimizer is None:
            optimizer = torch.optim.SGD(substitute_model.parameters(), lr=0.01,
                                        momentum=0.8)
        if loss is None:
            loss = F.cross_entropy

        victim_output_type = "labels"
        super().__init__(victim_model, substitute_model, optimizer, loss,
                         num_classes, victim_output_type, lr_scheduler)
        self.trainer_settings._validation = False

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="CopyCat attack")
        parser.add_argument("--training_epochs", default=5, type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 5)")

        cls._add_base_args(parser)

        return parser

    def _run(self):
        self._logger.info("########### Starting CopyCat attack ###########")
        self._logger.info(
                "CopyCat's attack budget: {}".format(len(self._thief_dataset)))

        # Get stolen labels from victim model
        self._logger.info("Getting stolen labels")
        stolen_labels = self._get_predictions(self._victim_model,
                                              self._thief_dataset)

        synthetic_dataset = CustomLabelDataset(self._thief_dataset,
                                               stolen_labels)

        self._logger.info("Training substitute model with synthetic dataset")
        self._train_substitute_model(synthetic_dataset)

        return
