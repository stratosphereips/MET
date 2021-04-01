import datetime
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from mef.utils.ios import mkdir_if_missing
from mef.utils.logger import set_up_logger
from mef.utils.pytorch.lighting.module import TrainableModel, VictimModel
from mef.utils.pytorch.lighting.trainer import get_trainer_with_settings
from mef.utils.settings import BaseSettings, TrainerSettings


class Base(ABC):
    attack_settings = None
    base_settings = BaseSettings()
    trainer_settings = TrainerSettings()

    def __init__(self, victim_model: VictimModel, substitute_model: TrainableModel):
        self._logger = None
        # Datasets
        self._test_set = None
        self._thief_dataset = None

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
            "--gpu", action="store_true", help="Whether to use gpu (Default: False)"
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
            help="Precision of caluclation in bits must be "
            "one of {16, 32} (Default: 32)",
        )
        parser.add_argument(
            "--accuracy",
            action="store_true",
            help="If accuracy should be used as metric for "
            "early stopping. If False F1-macro is used "
            "instead. (Default: False)",
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
    def _get_attack_parser(cls) -> ArgumentParser:
        pass

    @classmethod
    def get_attack_args(cls) -> ArgumentParser:
        parser = cls._get_attack_parser()
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
                pin_memory=self.base_settings.gpu,
                num_workers=self.base_settings.num_workers,
                shuffle=True,
                batch_size=self.base_settings.batch_size,
            )

        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(
                dataset=val_set,
                pin_memory=self.base_settings.gpu,
                num_workers=self.base_settings.num_workers,
                batch_size=self.base_settings.batch_size,
            )

        trainer, checkpoint_cb = get_trainer_with_settings(
            self.base_settings,
            self.trainer_settings,
            model_name="substitute",
            iteration=iteration,
            validation=val_set is not None,
        )

        trainer.fit(self._substitute_model, train_dataloader, val_dataloader)

        if not isinstance(checkpoint_cb, bool):
            # Load state dict of the best model from checkpoint
            checkpoint = torch.load(checkpoint_cb.best_model_path)
            self._substitute_model.load_state_dict(checkpoint["state_dict"])

        # For some reason the model after fit is on CPU and not GPU
        if self.base_settings.gpu:
            self._substitute_model.cuda()

        return

    def _test_model(
        self, model: Union[TrainableModel, VictimModel], test_set: Dataset
    ) -> float:
        test_dataloader = DataLoader(
            dataset=test_set,
            pin_memory=self.base_settings.gpu,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )

        trainer, _ = get_trainer_with_settings(
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
        torch.save(
            dict(state_dict=self._substitute_model.state_dict()), final_model_loc
        )

        return

    def _get_predictions(
        self, model: Union[TrainableModel, VictimModel], data: Dataset
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        model.eval()
        loader = DataLoader(
            dataset=data,
            pin_memory=self.base_settings.gpu,
            num_workers=self.base_settings.num_workers,
            batch_size=self.base_settings.batch_size,
        )
        hidden_layer_outputs = []
        y_hats = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Getting predictions"):
                output = model(x)
                y_hats.append(output[0].detach().cpu())
                if len(output) == 2:
                    hidden_layer_outputs.append(output[1].detach().cpu())

        y_hats = torch.cat(y_hats)

        if len(hidden_layer_outputs) != 0:
            return y_hats, torch.cat(hidden_layer_outputs)

        return y_hats

    def _finalize_attack(self) -> None:
        self._logger.info("########### Final attack metrics ###########")
        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()
        return

    # TODO: rework this
    def __call__(self, *args, **kwargs) -> None:
        """
        Starts the attack, the expected input is either (sub_dataset,
        test_set) or (test_set), where each parameter type is either
        TensorDataset or NumpyDataset.
        :param args:
        :param kwargs:
        :return: None
        """
        self._logger = set_up_logger(
            f"Mef",
            "debug" if self.base_settings.debug else "info",
            self.base_settings.save_loc,
        )

        # Seed random generators for reproducibility
        seed_everything(self.base_settings.seed)
        # In 1.7.0 still in BETA
        # torch.set_deterministic(self.base_settings.deterministic)

        # TODO: add self._device attribute + parameter to mefmodel
        if self.base_settings.gpu:
            self._victim_model.cuda()
            self._substitute_model.cuda()
            if hasattr(self, "_generator"):
                self._generator.cuda()

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
