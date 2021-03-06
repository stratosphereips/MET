import datetime
import time
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from ..utils.ios import mkdir_if_missing
from ..utils.logger import set_up_logger
from ..utils.pytorch.lightning.module import TrainableModel, VictimModel
from ..utils.pytorch.lightning.trainer import get_trainer_with_settings
from ..utils.settings import AttackSettings, BaseSettings, TrainerSettings


class AttackBase(ABC):
    attack_settings: AttackSettings

    def __init__(
        self,
        victim_model: VictimModel,
        substitute_model: TrainableModel,
        training_epochs: int = 1000,
        patience: Optional[int] = None,
        evaluation_frequency: Optional[int] = None,
        precision: int = 32,
        use_accuracy: bool = False,
        save_loc: Path = Path("./cache/"),
        gpu: Optional[int] = None,
        num_workers: int = 1,
        batch_size: int = 32,
        seed: Optional[int] = None,
        deterministic: bool = False,
        debug: bool = False,
    ):
        """Base class for all of the model extraction attacks. Contains settings for all the individual parts of the tool. And also helper methods for the attacks.

        Args:
            victim_model (VictimModel): Victim model, which is the target of the attack, wrapped inside the VictimModel class.
            substitute_model (TrainableModel): Substitue model, which the attack will train.
            training_epochs (int, optional): Number of training epochs for which the substitute model should be trained. Defaults to 1000.
            patience (int, optional): Patience for the early stopping during training of substiute model. If specified early stopping will be used. Defaults to None.
            evaluation_frequency (int, optional): Evalution frequency if validation set is available during training of a substitute model. Some attacks will automatically create validation set from adversary dataset if the user did not specify it himself. Defaults to None.
            precision (int, optional): Number precision that should be used. Currently only used in the pytorch-lightning trainer. Defaults to 32.
            use_accuracy (bool, optional): Whether to use accuracy during validation for checkpointing or F1-score, which is used by default. Defaults to False.
            save_loc (Path, optional): Location where log and other files created during the attack should be saved. Defaults to Path("./cache/").
            gpu (int, optional): Id of the gpu that should be used for the training. Defaults to None.
            num_workers (int, optional): Number of workers that should be used for data loaders. Defaults to 1.
            batch_size (int, optional): Batch size that should be used throughout the attack. Defaults to 32.
            seed (int, optional): Seed that should be used to initialize random generators, to help with reproducibility of results. Defaults to None.
            deterministic (bool, optional): Whether training should tried to be deterministic. Defaults to False.
            debug (bool, optional): Adds additional details to the log file and also performs all testing, training with only one batch. Defaults to False.
        """
        self.trainer_settings = TrainerSettings(
            training_epochs, patience, evaluation_frequency, precision, use_accuracy
        )
        self.base_settings = BaseSettings(
            save_loc, gpu, num_workers, batch_size, seed, deterministic, debug
        )
        self._logger: Logger
        # Datasets
        self._test_set: Dataset
        self._adversary_dataset: Dataset

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

    @classmethod
    def _add_base_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--seed", default=0, type=int, help="Random seed to be used (Default: 0)"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size to be used (Default: 32)",
        )
        parser.add_argument(
            "--save_loc",
            type=str,
            default="./cache/",
            help="Path where the attacks file should be saved (Default: ./cache/)",
        )
        # TODO: rework this so it supports multiple gpu and selection of gpu
        parser.add_argument(
            "--gpu",
            type=int,
            default=None,
            help="Whether to use gpu. If you want to use gpu write ID of the gpu that should be used. (Default: None)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="Number of workers to be used in loaders (Default: 1)",
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Run in deterministic mode (Default: False)",
        )
        parser.add_argument(
            "--debug", action="store_true", help="Run in debug mode (Default: False)"
        )
        parser.add_argument(
            "--precision",
            default=32,
            type=int,
            help="Precision of caluclation in bits must be one of {16, 32} (Default: 32)",
        )
        parser.add_argument(
            "--accuracy",
            action="store_true",
            help="If accuracy should be used as metric for early stopping. If False F1-macro is used instead. (Default: False)",
        )
        return

    @classmethod
    def _add_trainer_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--training_epochs",
            default=100,
            type=int,
            help="Number of training epochs for substitute model (Default: 100)",
        )
        parser.add_argument(
            "--patience",
            default=10,
            type=int,
            help="Number of epochs without improvement for early stop (Default: 10)",
        )
        parser.add_argument(
            "--evaluation_frequency",
            default=1,
            type=int,
            help="Epochs interval of validation (Default: 1)",
        )

        return

    @classmethod
    @abstractmethod
    def _get_attack_parser(
        cls, parser: Optional[ArgumentParser] = None
    ) -> ArgumentParser:
        pass

    @classmethod
    def get_attack_args(cls, parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        """Creates ArgumentParser with the attack's and tool's parameters. If existing parser is passed to this method, the parameters will be added to it.

        Args:
            parser (Optional[ArgumentParser], optional): Existing parser to which the parameters should be added. Defaults to None.

        Returns:
            ArgumentParser: Parser containing the parameters of the attack and tool.
        """
        parser = cls._get_attack_parser(parser)
        cls._add_base_args(parser)
        cls._add_trainer_args(parser)

        return parser

    def _train_substitute_model(
        self,
        train_set: Dataset,
        val_set: Optional[Dataset] = None,
        iteration: Optional[int] = None,
    ) -> None:
        # For ripper attack
        if isinstance(train_set, IterableDataset):
            train_dataloader = DataLoader(dataset=train_set)
        else:
            train_dataloader = DataLoader(
                dataset=train_set,
                pin_memory=True if self.base_settings.gpu is not None else False,
                num_workers=self.base_settings.num_workers,
                shuffle=True,
                batch_size=self.base_settings.batch_size,
            )

        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(
                dataset=val_set,
                pin_memory=True if self.base_settings.gpu is not None else False,
                num_workers=self.base_settings.num_workers,
                batch_size=self.base_settings.batch_size,
            )

        trainer = get_trainer_with_settings(
            self.base_settings,
            self.trainer_settings,
            model_name="substitute",
            iteration=iteration,
            validation=val_set is not None,
        )
        # Ignore warning with missing validation dataloader from pytorch-lightning, while val_step is implemented in TrainingModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(self._substitute_model, train_dataloader, val_dataloader)

        if trainer.checkpoint_callback is not None:
            self._logger.info("Loading best training checkpoint of substitute model!")
            self._substitute_model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )
            # TODO: workout the reason why the load_from_checkpoint is not loading the correct instance of the model
            checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
            self._substitute_model.model = checkpoint["hyper_parameters"]["model"]
            if self.base_settings.gpu is not None:
                self._substitute_model.cuda(self.base_settings.gpu)

        return

    def _test_model(
        self, model: Union[TrainableModel, VictimModel], test_set: Dataset
    ) -> float:
        test_dataloader = DataLoader(
            dataset=test_set,
            pin_memory=True if self.base_settings.gpu is not None else False,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        trainer = get_trainer_with_settings(
            self.base_settings, self.trainer_settings, logger=False
        )
        metrics = trainer.test(model, test_dataloader)

        return 100 * metrics[0]["test_acc"]

    def _get_test_set_metrics(self) -> None:
        self._logger.info("Test set metrics")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)
        sub_test_acc = self._test_model(self._substitute_model, self._test_set)
        self._logger.info("Victim model Accuracy: {:.1f}%".format(vict_test_acc))
        self._logger.info("Substitute model Accuracy: {:.1f}%".format(sub_test_acc))

        return

    def _get_aggreement_score(self) -> None:
        vict_test_labels = self._victim_model.test_labels
        sub_test_labels = self._substitute_model.test_labels

        agreement_count = np.sum((vict_test_labels == sub_test_labels))
        self._logger.info(
            "Agreement score: {}/{} ({:.1f}%)".format(
                agreement_count,
                len(vict_test_labels),
                100 * (agreement_count / len(vict_test_labels)),
            )
        )

        return

    def _save_final_subsitute(self) -> None:
        final_model_dir = self.base_settings.save_loc.joinpath("substitute")
        mkdir_if_missing(final_model_dir)
        final_model_loc = final_model_dir.joinpath(
            "final_substitute_model-state_dict.pt"
        )
        self._logger.info(
            "Saving final substitute model state dictionary to: {}".format(
                final_model_loc.__str__()
            )
        )
        self._substitute_model.save(final_model_loc)

        return

    def _get_predictions(
        self, model: Union[TrainableModel, VictimModel], data: Dataset
    ) -> torch.Tensor:
        model.eval()
        loader = DataLoader(
            dataset=data,
            pin_memory=True if self.base_settings.gpu is not None else False,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        y_hats = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Getting predictions"):
                output = model(x)
                y_hats.append(output[0].detach().cpu())

        return torch.cat(y_hats)

    def _get_embeddings(
        self, model: Union[TrainableModel, VictimModel], data: Dataset
    ) -> torch.Tensor:
        model.eval()
        loader = DataLoader(
            dataset=data,
            pin_memory=True if self.base_settings.gpu is not None else False,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        embeddings = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Getting predictions"):
                output = model(x)
                embeddings.append(output[1].detach().cpu())

        return torch.cat(embeddings)

    def _finalize_attack(self) -> None:
        self._logger.info("########### Final attack metrics ###########")
        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()
        return

    # TODO: rework this
    def __call__(self, *args: Dataset, **kwargs: Dataset) -> None:
        """
        Starts the attack, the expected input is either (sub_dataset,
        test_set) or (test_set), where each parameter type is either
        TensorDataset or NumpyDataset.
        :param args:
        :param kwargs:
        :return: None
        """
        self._logger = set_up_logger(
            "Met",
            "debug" if self.base_settings.debug else "info",
            self.base_settings.save_loc,
        )

        # Seed random generators for reproducibility
        seed_everything(self.base_settings.seed)
        # In 1.7.0 still in BETA
        # torch.set_deterministic(self.base_settings.deterministic)

        if self.base_settings.gpu is not None:
            self._victim_model.cuda(self.base_settings.gpu)
            self._substitute_model.cuda(self.base_settings.gpu)
            if hasattr(self, "_generator"):
                self._generator.cuda(self.base_settings.gpu)

        start_time = time.time()
        self._run(*args, **kwargs)
        end_time = time.time()

        self._finalize_attack()
        final_time = str(datetime.timedelta(seconds=end_time - start_time))
        self._logger.info(f"Attacks's time: {final_time}")

        # Close all handlers of logger
        for i in list(self._logger.handlers):
            self._logger.removeHandler(i)
            i.flush()
            i.close()

        return

    @abstractmethod
    def _run(self, *args, **kwargs) -> None:
        pass
