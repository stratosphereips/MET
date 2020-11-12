import argparse

import torch
import torch.nn.functional as F

from mef.utils.pytorch.datasets import CustomLabelDataset
from .base import Base


class CopyCat(Base):

    def __init__(self, victim_model, substitute_model):
        optimizer = torch.optim.SGD(substitute_model.parameters(), lr=0.01,
                                    momentum=0.8)
        loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss)
        self.trainer_settings.validation = False

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="CopyCat attack")
        cls._add_base_args(parser)

        return parser

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info("########### Starting CopyCat attack ###########")
        self._logger.info(
                "CopyCat's attack budget: {}".format(len(self._thief_dataset)))

        # Get stolen labels from victim model
        self._logger.info("Getting stolen labels")
        stolen_labels = self._get_predictions(self._victim_model,
                                              self._thief_dataset,
                                              "labels")

        synthetic_dataset = CustomLabelDataset(self._thief_dataset,
                                               stolen_labels)

        self._logger.info("Training substitute model with synthetic dataset")
        self._train_substitute_model(synthetic_dataset)

        self._finalize_attack()

        return
