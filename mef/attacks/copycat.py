from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from ..utils.pytorch.datasets import CustomLabelDataset, split_dataset
from ..utils.pytorch.functional import get_class_labels
from ..utils.pytorch.lighting.module import TrainableModel, VictimModel
from .base import AttackBase


class CopyCat(AttackBase):
    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        *args: Union[int, bool, Path],
        **kwargs: Union[int, bool, Path]
    ):
        super().__init__(victim_model, substitute_model, *args, **kwargs)

    @classmethod
    def _get_attack_parser(cls) -> ArgumentParser:
        return ArgumentParser(description="CopyCat attack")

    def _check_args(self, sub_data: Dataset, test_set: Dataset) -> None:
        if not isinstance(sub_data, Dataset):
            self._logger.error("Substitute dataset must be Pytorch's dataset.")
            raise TypeError()
        if not isinstance(test_set, Dataset):
            self._logger.error("Test set must be Pytorch's dataset.")
            raise TypeError()

        self._thief_dataset = sub_data
        self._test_set = test_set

        return

    def _run(self, sub_data: Dataset, test_set: Dataset) -> None:
        self._check_args(sub_data, test_set)
        self._logger.info("########### Starting CopyCat attack ###########")
        self._logger.info(
            "CopyCat's attack budget: {}".format(len(self._thief_dataset))
        )

        # Get stolen labels from victim model
        self._logger.info("Getting stolen labels")
        stolen_labels = self._get_predictions(self._victim_model, self._thief_dataset)

        synthetic_dataset = CustomLabelDataset(self._thief_dataset, stolen_labels)

        train_set = synthetic_dataset
        val_set = None
        if self.trainer_settings.evaluation_frequency is not None:
            train_set, val_set = split_dataset(train_set, 0.2)

        self._logger.info("Training substitute model with synthetic dataset")
        self._train_substitute_model(train_set, val_set)

        return
