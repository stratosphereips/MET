from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.logger import set_up_logger
from mef.utils.pytorch.datasets import CustomDataset, MefDataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.trainer import get_trainer_with_settings
from mef.utils.settings import BaseSettings, DataSettings, TrainerSettings


class Base(ABC):
    attack_settings = None
    base_settings = BaseSettings()
    data_settings = DataSettings()
    trainer_settings = TrainerSettings()

    def __init__(self,
                 victim_model,
                 substitute_model,
                 optimizer,
                 loss,
                 num_classes,
                 lr_scheduler=None):
        self._logger = None
        # Datasets
        self._test_set = None
        self._thief_dataset = None
        self._num_classes = num_classes

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        if self.base_settings.gpus is not None:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        # Optimizer, loss_functions and lr_scheduler for substitute model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._loss = loss

    @classmethod
    @abstractmethod
    def get_attack_args(cls):
        pass

    @classmethod
    def _add_base_args(cls, parser):
        parser.add_argument("--seed", default=0, type=int,
                            help="Random seed to be used (Default: 0)")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Batch size to be used (Default: 64)")
        parser.add_argument("--save_loc", type=str, default="./cache/",
                            help="Path where the attacks file should be "
                                 "saved (Default: ./cache/)")
        parser.add_argument("--gpus", type=int, default=0,
                            help="Number of gpus to be used (Default: 0)")
        parser.add_argument("--deterministic", action="store_false",
                            help="Run in deterministic mode (Default: True)")
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
        dataset = MefDataset(self.data_settings.batch_size, train_set, val_set)
        train_dataloader = dataset.train_dataloader()

        val_dataloader = None
        if dataset.val_set is not None:
            val_dataloader = dataset.val_dataloader()

        trainer = get_trainer_with_settings(self.base_settings,
                                            self.trainer_settings,
                                            "substitute", iteration,
                                            dataset.val_set is None)

        mef_model = MefModule(self._substitute_model, self._num_classes,
                              self._optimizer, self._loss, self._lr_scheduler)
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        # For some reason the model after fit is on CPU and not GPU
        if self.base_settings.gpus is not None:
            self._substitute_model.cuda()

        return

    def _test_model(self,
                    model,
                    test_set):
        test_set = MefDataset(self.data_settings.batch_size, test_set=test_set)
        test_dataloader = test_set.test_dataloader()
        mef_model = MefModule(model, self._num_classes, loss=self._loss)

        trainer = get_trainer_with_settings(self.base_settings,
                                            self.trainer_settings,
                                            model_name='', iteration=None,
                                            validation=False)
        metrics = trainer.test(mef_model, test_dataloader)

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
        self._logger.info("Getting attack metric")
        # Agreement score
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=-1).numpy()
        sub_test_labels = self._get_predictions(self._substitute_model,
                                                self._test_set)
        sub_test_labels = torch.argmax(sub_test_labels, dim=-1).numpy()

        agreement_count = np.sum(vict_test_labels == sub_test_labels)
        self._logger.info("Agreement count: {}".format(agreement_count))
        self._logger.info(
                "Test agreement between victim and substitute model on test "
                "dataset {:.1f}%"
                    .format(100 * (agreement_count / len(vict_test_labels))))

        return

    def _save_final_subsitute(self):
        final_model_loc = self.base_settings.save_loc.joinpath(
                "final_substitute_model.pt")
        self._logger.info(
                "Saving final substitute model state dictionary to: {}".format(
                        final_model_loc.__str__()))
        torch.save(dict(state_dict=self._substitute_model.state_dict()),
                   final_model_loc)

        return

    def _get_predictions(self,
                         model,
                         data,
                         output_type="softmax",
                         return_all_layers=False):
        if output_type not in ["one_hot", "softmax", "logits", "labels"]:
            self._logger.error(
                    "Output_type must be one of {one_hot, softmax, logits, "
                    "labels}")
            raise ValueError()

        model.eval()
        data = MefDataset(self.data_settings.batch_size, val_set=data)
        loader = data.val_dataloader()
        hidden_layer_outputs = []
        y_preds = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Getting predictions"):
                if return_all_layers:
                    y_pred, feature_vec = model(x, return_all_layers=True)
                    hidden_layer_outputs.append(feature_vec.detach().cpu())
                else:
                    y_pred = model(x)
                y_preds.append(y_pred.cpu())

        y_preds = torch.cat(y_preds)
        if return_all_layers:
            hidden_layer_outputs = torch.cat(hidden_layer_outputs)

        if output_type == "one_hot":
            y_hat = F.one_hot(torch.argmax(y_preds, dim=-1),
                              num_classes=y_preds.size()[-1])
            # to_oneshot returns tensor with uint8 type
            y_hat = y_hat.float()
        elif output_type == "softmax":
            y_hat = F.softmax(y_preds, dim=-1)
        elif output_type == "labels":
            y_hat = torch.argmax(y_preds, dim=-1)
        else:
            y_hat = y_preds

        if return_all_layers:
            return y_hat, hidden_layer_outputs

        return y_hat

    def _parse_args(self, args, kwargs):
        # Numpy input (x_sub, y_sub, x_test, y_test)
        try:
            if len(args) == 4:
                for arg in args:
                    if not isinstance(arg, np.ndarray):
                        raise TypeError()
                else:
                    self._thief_dataset = CustomDataset(args[0], args[1])
                    self._test_set = CustomDataset(args[2], args[3])
                    return
            elif len(kwargs) == 4:
                for _, value in kwargs.items():
                    if not isinstance(value, np.ndarray):
                        raise TypeError()
                else:
                    if "x_sub" not in kwargs:
                        self._logger.error("x_sub input argument is missing")
                        raise ValueError()
                    if "y_sub" not in kwargs:
                        self._logger.error("y_sub input argument is missing")
                        raise ValueError()
                    if "x_test" not in kwargs:
                        self._logger.error("x_test input argument is missing")
                        raise ValueError()
                    if "y_test" not in kwargs:
                        self._logger.error("y_test input argument is missing")
                        raise ValueError()

                    self._thief_dataset = CustomDataset(kwargs["x_sub"],
                                                        kwargs["y_sub"])
                    self._test_set = CustomDataset(kwargs["x_test"],
                                                   kwargs["y_test"])
                    return
            elif len(args) == 2:
                # Pytorch input (sub_dataset, test_set)
                for arg in args:
                    if not isinstance(arg, torch.utils.data.Dataset):
                        break
                else:
                    self._thief_dataset = args[0]
                    self._test_set = args[1]
                    return
                # Pytorch input (x_test, y_test)
                for arg in args:
                    if not isinstance(arg, np.ndarray):
                        break
                else:
                    self._thief_dataset = CustomDataset(args[0], args[1])
                    return
                TypeError()
            elif len(kwargs) == 2:
                # Pytorch input (sub_dataset, test_set)
                for _, value in kwargs.items():
                    if not isinstance(value, torch.utils.data.Dataset):
                        break
                else:
                    if "sub_dataset" not in kwargs:
                        self._logger.error(
                                "sub_dataset input argument is missing")
                        raise ValueError()
                    if "test_set" not in kwargs:
                        self._logger.error(
                                "test_set input argument is missing")
                        raise ValueError()

                    self._thief_dataset = kwargs["sub_dataset"]
                    self._test_set = kwargs["test_set"]
                    return
                # Pytorch input (x_test, y_test)
                for _, value in kwargs.items():
                    if not isinstance(value, np.ndarray):
                        break
                else:
                    if "x_test" not in kwargs:
                        self._logger.error(
                                "x_test input argument is missing")
                        raise ValueError()
                    if "y_test" not in kwargs:
                        self._logger.error(
                                "y_test input argument is missing")
                        raise ValueError()

                    self._test_set = CustomDataset(kwargs["x_test"],
                                                   kwargs["y_test"])
                    return
                TypeError()
            elif len(args) == 1:
                if not isinstance(args[0], torch.utils.data.Dataset):
                    TypeError()
                self._test_set = args[0]
            elif len(kwargs) == 1:
                if not isinstance(kwargs["test_set"],
                                  torch.utils.data.Dataset):
                    TypeError()
                if "test_set" not in kwargs:
                    self._logger.error(
                            "test_set input argument is missing")
                    raise ValueError()
                self._test_set = kwargs["test_set"]
        # Pytorch input (test_set)

        except ValueError or TypeError:
            self._logger.error(
                    "Input arguments for attack must be either numpy arrays ("
                    "x_sub, y_sub, x_test, y_test), (x_test, y_test) or "
                    "Pytorch datasets (sub_dataset, test_set), (test_set)")
            exit()
        return

    def _finalize_attack(self):
        self._logger.info("########### Final attack metrics ###########")
        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()
        return

    def __call__(self, *args, **kwargs):
        # Seed random generators for reproducibility
        seed_everything(self.base_settings.seed)
        # In 1.7.0 still in BETA
        # torch.set_deterministic(self.base_settings.deterministic)

        self._logger = set_up_logger("Mef",
                                     "debug" if self.base_settings.debug else
                                     "info", self.base_settings.save_loc)
        self._parse_args(args, kwargs)
        self._run()
        self._finalize_attack()

        return

    @abstractmethod
    def _run(self):
        pass
