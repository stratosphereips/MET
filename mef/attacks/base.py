import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import to_onehot
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.pytorch.datasets import CustomDataset, NoYDataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer


class Base:
    _test_config = None
    _logger = None

    def __init__(self, victim_model, substitute_model, x_test, y_test,
                 optimizer, train_loss, test_loss, training_epochs=100,
                 early_stop_tolerance=10, evaluation_frequency=2,
                 val_size=0.2, batch_size=64, num_classes=None,
                 save_loc="./cache", validation=True):
        self._val_size = val_size
        self._batch_size = batch_size
        self._save_loc = save_loc

        self._trainer_kwargs = dict(
                gpus=self._test_config.gpus,
                training_epochs=training_epochs,
                early_stop_tolerance=early_stop_tolerance,
                evaluation_frequency=evaluation_frequency,
                save_loc=self._save_loc,
                debug=self._test_config.debug,
                deterministic=self._test_config.deterministic,
                validation=validation
        )

        # Test set
        self._test_set = CustomDataset(x_test, y_test)

        self._num_classes = num_classes

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        if self._test_config.gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

        # Optimizer, loss_functions
        self._optimizer = optimizer
        self._train_loss = train_loss
        self._test_loss = test_loss

    def _train_model(self, model, optimizer, loss, train_set, val_set=None,
                     iteration=None, worker_init_fn=None,
                     training_epochs=None):
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

        trainer_kwargs = self._trainer_kwargs
        if training_epochs is not None:
            trainer_kwargs["training_epochs"] = training_epochs
        trainer = get_trainer(**self._trainer_kwargs, iteration=iteration)

        mef_model = MefModule(model, optimizer, loss)
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        # For some reason the model after fit is on CPU and not GPU
        if self._test_config.gpus:
            model.cuda()

        return

    def _test_model(self, model, loss, test_set):
        test_dataloader = DataLoader(dataset=test_set, pin_memory=True,
                                     num_workers=4,
                                     batch_size=self._batch_size)
        mef_model = MefModule(model, loss=loss)
        trainer = get_trainer(**self._trainer_kwargs)
        metrics = trainer.test(mef_model, test_dataloader)

        return 100 * metrics[0]["test_acc"], metrics[0]["test_loss"]

    def _get_aggreement_score(self):
        self._logger.info("Getting attack metric")
        # Agreement score
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set.x)
        vict_test_labels = np.argmax(vict_test_labels, axis=1)
        sub_test_labels = self._get_predictions(self._substitute_model,
                                                self._test_set.x)
        sub_test_labels = np.argmax(sub_test_labels, axis=1)

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
        torch.save(dict(state_dict=self._substitute_model.state_dict),
                   final_model_loc)

        return

    def _get_predictions(self, model, x, output_type="softmax"):
        model.eval()

        data = NoYDataset(x)
        loader = DataLoader(data, pin_memory=True, num_workers=4,
                            batch_size=self._batch_size)
        y_preds = []
        with torch.no_grad():
            for _, x in enumerate(tqdm(loader, desc="Getting predictions",
                                       total=len(loader))):
                y_pred = model(x)
                y_preds.append(y_pred.cpu())

        y_preds = torch.cat(y_preds)

        if output_type == "one_hot":
            y_hat = to_onehot(torch.argmax(y_preds, dim=1),
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

        return y_hat.numpy()

    def run(self, x, y):
        raise NotImplementedError("Attacks must implement run method!")
