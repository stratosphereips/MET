from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Optional

from torch.utils.data import Dataset

from ..utils.pytorch.datasets import CustomLabelDataset, split_dataset
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
        """Implementation of CopyCat attack from .

        Args:
            victim_model (VictimModel): Victim model, which is the target of the attack, wrapped inside the VictimModel class.
            substitute_model (TrainableModel): Substitue model, which the attack will train.
            training_epochs (int, optional): Number of training epochs for which the substitute model should be trained. Defaults to 1000.
            patience (int, optional): Patience for the early stopping during training of substiute model. If specified early stopping will be used. Defaults to None.
            evaluation_frequency (int, optional): Evalution frequency if validation set is available during training of a substitute model. Some attacks will automatically create validation set from adversary dataset if the user did not specify it himself. Defaults to None.
            precision (int, optional): Number precision that should be used. Currently only used in the pytorch-lighting trainer. Defaults to 32.
            use_accuracy (bool, optional): Whether to use accuracy during validation for checkpointing or F1-score, which is used by default. Defaults to False.
            save_loc (Path, optional): Location where log and other files created during the attack should be saved. Defaults to Path("./cache/").
            gpu (int, optional): Id of the gpu that should be used for the training. Defaults to None.
            num_workers (int, optional): Number of workers that should be used for data loaders. Defaults to 1.
            batch_size (int, optional): Batch size that should be used throughout the attack. Defaults to 32.
            seed (int, optional): Seed that should be used to initialize random generators, to help with reproducibility of results. Defaults to None.
            deterministic (bool, optional): Whether training should tried to be deterministic. Defaults to False.
            debug (bool, optional): Adds additional details to the log file and also performs all testing, training with only one batch. Defaults to False.
        """
        super().__init__(victim_model, substitute_model, *args, **kwargs)

    @classmethod
    def _get_attack_parser(
        cls, parser: Optional[ArgumentParser] = None
    ) -> ArgumentParser:
        return ArgumentParser(description="CopyCat attack") if parser is None else parser

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
