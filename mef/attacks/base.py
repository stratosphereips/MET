import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.logger import set_up_logger
from mef.utils.pytorch.datasets import CustomDataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer


class Base:

    def __init__(self, victim_model, substitute_model, optimizer,
                 loss, lr_scheduler=None, training_epochs=100,
                 early_stop_tolerance=10, evaluation_frequency=2,
                 val_size=0.2, batch_size=64, num_classes=None,
                 save_loc="./cache", validation=True, gpus=0, seed=None,
                 deterministic=True, debug=False, precision=32):
        # Mef set up
        self._gpus = gpus
        self._save_loc = save_loc
        seed_everything(seed)
        self._logger = set_up_logger("Mef", "debug" if debug else "info",
                                     self._save_loc)

        # Datasets
        self._test_set = None
        self._thief_dataset = None
        self._val_size = val_size
        self._batch_size = batch_size
        self._num_classes = num_classes

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        if self._gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        # Optimizer, loss_functions
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._loss = loss

        # Pytorch-lighting trainer
        self._trainer_kwargs = dict(
                gpus=self._gpus,
                training_epochs=training_epochs,
                early_stop_tolerance=early_stop_tolerance,
                evaluation_frequency=evaluation_frequency,
                save_loc=self._save_loc,
                debug=debug,
                deterministic=deterministic,
                validation=validation,
                precision=precision
        )

    def _train_model(self, model, optimizer, train_set, val_set=None,
                     iteration=None, worker_init_fn=None,
                     training_epochs=None, lr_scheduler=None):
        train_dataloader = DataLoader(dataset=train_set, pin_memory=True,
                                      num_workers=4,
                                      batch_size=self._batch_size,
                                      worker_init_fn=worker_init_fn)
        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                        num_workers=4,
                                        batch_size=self._batch_size,
                                        worker_init_fn=worker_init_fn)

        trainer_kwargs = self._trainer_kwargs.copy()
        if training_epochs is not None:
            trainer_kwargs["training_epochs"] = training_epochs
        trainer = get_trainer(**trainer_kwargs, iteration=iteration,
                              accuracy=False)

        mef_model = MefModule(model, self._num_classes, optimizer, self._loss,
                              lr_scheduler)
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        # For some reason the model after fit is on CPU and not GPU
        if self._gpus:
            model.cuda()

        return

    def _test_model(self, model, test_set):
        test_dataloader = DataLoader(dataset=test_set, pin_memory=True,
                                     num_workers=4,
                                     batch_size=self._batch_size)
        mef_model = MefModule(model, self._num_classes, loss=self._loss)
        trainer = get_trainer(**self._trainer_kwargs)
        metrics = trainer.test(mef_model, test_dataloader)

        return 100 * metrics[0]["test_acc"]

    def _get_test_set_metrics(self):
        self._logger.info("Test set metrics")
        vict_test_acc = self._test_model(self._victim_model, self._test_set)
        sub_test_acc = self._test_model(self._substitute_model, self._test_set)
        self._logger.info(
                "Victim model F1-score: {:.3f}".format(vict_test_acc))
        self._logger.info(
                "Substitute model F1-score: {:.3f}".format(sub_test_acc))

        return

    def _get_aggreement_score(self):
        self._logger.info("Getting attack metric")
        # Agreement score
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=1).numpy()
        sub_test_labels = self._get_predictions(self._substitute_model,
                                                self._test_set)
        sub_test_labels = torch.argmax(sub_test_labels, dim=1).numpy()

        agreement_count = np.sum(vict_test_labels == sub_test_labels)
        self._logger.info("Agreement count: {}".format(agreement_count))
        self._logger.info(
                "Test agreement between victim and substitute model on test "
                "dataset {:.1f}%"
                    .format(100 * (agreement_count / len(vict_test_labels))))

        return

    def _save_final_subsitute(self):
        final_model_loc = self._save_loc + "/final_substitute_model.pt"
        self._logger.info(
                "Saving final substitute model state dictionary to: {}".format(
                        final_model_loc))
        torch.save(dict(state_dict=self._substitute_model.state_dict()),
                   final_model_loc)

        return

    def _get_predictions(self, model, data, output_type="softmax"):
        model.eval()
        loader = DataLoader(data, pin_memory=True, num_workers=4,
                            batch_size=self._batch_size)
        y_preds = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Getting predictions", total=len(
                    loader)):
                y_pred = model(x)
                y_preds.append(y_pred.cpu())

        y_preds = torch.cat(y_preds)

        if output_type == "one_hot":
            y_hat = F.one_hot(torch.argmax(y_preds, dim=1),
                              num_classes=y_preds.size()[1])
            # to_oneshot returns tensor with uint8 type
            y_hat = y_hat.float()
        elif output_type == "softmax":
            y_hat = F.softmax(y_preds, dim=1)
        elif output_type == "logits":
            y_hat = y_preds
        else:
            self._logger.error(
                    "Model output type must be one of {one_hot, softmax, "
                    "labels, logits}")
            raise ValueError()

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
            # Pytorch input (sub_dataset, test_set)
            elif len(args) == 2:
                for arg in args:
                    if not isinstance(arg, torch.utils.data.Dataset):
                        raise TypeError()
                else:
                    self._thief_dataset = args[0]
                    self._test_set = args[1]
            elif len(kwargs) == 2:
                for _, value in kwargs.items():
                    if not isinstance(value, torch.utils.data.Dataset):
                        raise TypeError()
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
        except ValueError or TypeError:
            self._logger.error(
                    "Input arguments for attack must be either numpy arrays ("
                    "x_sub, y_sub, x_test, y_test) or Pytorch datasets ("
                    "sub_dataset, test_set)")
            exit()
        return

    def run(self, *args, **kwargs):
        raise NotImplementedError("Attacks must implement run method!")
