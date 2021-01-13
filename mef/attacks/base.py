from abc import ABC, abstractmethod

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.logger import set_up_logger
from mef.utils.pytorch.lighting.trainer import get_trainer_with_settings
from mef.utils.settings import BaseSettings, TrainerSettings


class Base(ABC):
    attack_settings = None
    base_settings = BaseSettings()
    trainer_settings = TrainerSettings()

    def __init__(self,
                 victim_model,
                 substitute_model):
        self._logger = None
        # Datasets
        self._test_set = None
        self._thief_dataset = None

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

    @classmethod
    @abstractmethod
    def get_attack_args(cls):
        pass

    @classmethod
    def _add_base_args(cls, parser):
        parser.add_argument("--seed", default=0, type=int,
                            help="Random seed to be used (Default: 0)")
        parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size to be used (Default: 32)")
        parser.add_argument("--save_loc", type=str, default="./cache/",
                            help="Path where the attacks file should be "
                                 "saved (Default: ./cache/)")
        parser.add_argument("--gpus", type=int, default=0,
                            help="Number of gpus to be used (Default: 0)")
        parser.add_argument("--num_workers", type=int, default=1,
                            help="Number of workers to be used in loaders ("
                                 "Default: 1)")
        parser.add_argument("--deterministic", action="store_true",
                            help="Run in deterministic mode (Default: False)")
        parser.add_argument("--debug", action="store_true",
                            help="Run in debug mode (Default: False)")
        parser.add_argument("--precision", default=32, type=int,
                            help="Precision of caluclation in bits must be "
                                 "one of {16, 32} (Default: 32)")
        parser.add_argument("--accuracy", action="store_true",
                            help="If accuracy should be used as metric for "
                                 "early stopping. If False F1-macro is used "
                                 "instead. (Default: False)")
        return

    def _train_substitute_model(self,
                                train_set,
                                val_set=None,
                                iteration=None):
        train_dataloader = DataLoader(
                dataset=train_set, pin_memory=self.base_settings.gpus != 0,
                num_workers=self.base_settings.num_workers, shuffle=True,
                batch_size=self.base_settings.batch_size)

        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(
                    dataset=val_set, pin_memory=self.base_settings.gpus != 0,
                    num_workers=self.base_settings.num_workers,
                    batch_size=self.base_settings.batch_size)

        trainer, checkpoint_cb = get_trainer_with_settings(
                self.base_settings, self.trainer_settings,
                model_name="substitute", iteration=iteration,
                validation=val_set is not None)

        trainer.fit(self._substitute_model, train_dataloader, val_dataloader)

        if not isinstance(checkpoint_cb, bool):
            # Load state dict of the best model from checkpoint
            checkpoint = torch.load(checkpoint_cb.best_model_path)
            self._substitute_model.load_state_dict(checkpoint["state_dict"])

        # For some reason the model after fit is on CPU and not GPU
        if self.base_settings.gpus:
            self._substitute_model.cuda()

        return

    def _test_model(self,
                    model,
                    test_set):
        test_dataloader = DataLoader(
                dataset=test_set, pin_memory=self.base_settings.gpus != 0,
                num_workers=self.base_settings.num_workers,
                batch_size=self.base_settings.batch_size)

        trainer, _ = get_trainer_with_settings(self.base_settings,
                                               self.trainer_settings,
                                               logger=False)
        metrics = trainer.test(model, test_dataloader)

        return 100 * metrics[0]["test_acc"]

    def _get_test_set_metrics(self):
        self._logger.info("Test set metrics")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)
        sub_test_acc = self._test_model(self._substitute_model, self._test_set)
        self._logger.info(
                "Victim model Accuracy: {:.1f}%".format(vict_test_acc))
        self._logger.info(
                "Substitute model Accuracy: {:.1f}%".format(sub_test_acc))

        return

    def _get_aggreement_score(self):
        vict_test_labels = self._victim_model.test_labels
        sub_test_labels = self._substitute_model.test_labels

        agreement_count = np.sum((vict_test_labels == sub_test_labels))
        self._logger.info("Agreement score: {}/{} ({:.1f}%)".format(
                agreement_count, len(vict_test_labels),
                100 * (agreement_count / len(vict_test_labels))))

        return

    def _save_final_subsitute(self):
        final_model_loc = self.base_settings.save_loc.joinpath("substitute",
                "final_substitute_model-state_dict.pt")
        self._logger.info(
                "Saving final substitute model state dictionary to: {}".format(
                        final_model_loc.__str__()))
        torch.save(dict(state_dict=self._substitute_model.state_dict()),
                   final_model_loc)

        return

    def _get_predictions(self,
                         model,
                         data):
        model.eval()
        loader = DataLoader(dataset=data,
                            pin_memory=self.base_settings.gpus != 0,
                            num_workers=self.base_settings.num_workers,
                            batch_size=self.base_settings.batch_size)
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

    def _finalize_attack(self):
        self._logger.info("########### Final attack metrics ###########")
        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()
        return

    def __call__(self, *args, **kwargs):
        """
        Starts the attack, the expected input is either (sub_dataset,
        test_set) or (test_set), where each parameter type is either
        TensorDataset or NumpyDataset.
        :param args:
        :param kwargs:
        :return: None
        """
        # Seed random generators for reproducibility
        seed_everything(self.base_settings.seed)
        # In 1.7.0 still in BETA
        # torch.set_deterministic(self.base_settings.deterministic)

        self._logger = set_up_logger("Mef",
                                     "debug" if self.base_settings.debug else
                                     "info", self.base_settings.save_loc)

        # TODO: add self._device attribute + parameter to mefmodel
        if self.base_settings.gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        self._run(*args, **kwargs)
        self._finalize_attack()

        return

    @abstractmethod
    def _run(self, *args, **kwargs):
        pass
