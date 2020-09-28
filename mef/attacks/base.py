import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import to_onehot
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer


class Base:
    _test_config = None
    _logger = None

    def __init__(self, training_epochs, save_loc):
        self._save_loc = save_loc
        trainer_kargs = dict(gpus=self._test_config.gpus,
                             training_epochs=training_epochs,
                             early_stop_tolerance=self._test_config.early_stop_tolerance,
                             evaluation_frequency=self._test_config.evaluation_frequency,
                             save_loc=self._save_loc,
                             debug=self._test_config.debug)
        self._trainer = get_trainer(**trainer_kargs)

    def _train_model(self, model, optimizer, loss, train_set, val_set=None):
        train_dataloader = DataLoader(dataset=train_set, pin_memory=True,
                                      num_workers=4,
                                      batch_size=self._test_config.batch_size)
        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                        num_workers=4,
                                        batch_size=self._test_config.batch_size)

        mef_model = MefModule(model, optimizer, loss)
        self._trainer.fit(mef_model, train_dataloader, val_dataloader)

        return

    def _test_model(self, model, loss, test_set, labels=True):
        test_dataloader = DataLoader(dataset=test_set, pin_memory=True,
                                     num_workers=4,
                                     batch_size=self._test_config.batch_size)
        mef_model = MefModule(model, loss=loss, labels=labels)
        metrics = self._trainer.test(mef_model, test_dataloader)

        return metrics[0]["test_acc"], metrics[0]["test_loss"]

    def _get_attack_metric(self, substitute_model, test_dataset,
                           vict_test_labels):
        # Agreement score
        sub_test_labels = self._get_predictions(substitute_model, test_dataset)
        sub_test_labels = torch.argmax(sub_test_labels, dim=1).numpy()

        agreement_count = np.sum(vict_test_labels == sub_test_labels)
        self._logger.info("Agreement count: {}".format(agreement_count))
        self._logger.info(
                "Test agreement between victim and substitute model on test "
                "dataset {:.1f}%"
                    .format(100 * (agreement_count / len(vict_test_labels))))

        return

    def _get_predictions(self, model, data, output_type="softmax"):
        model.eval()
        loader = DataLoader(data, pin_memory=True,
                            batch_size=self._test_config.batch_size,
                            num_workers=4)
        y_preds = []
        with torch.no_grad():
            for _, batch in enumerate(
                    tqdm(loader, desc="Getting predictions")):
                x, _ = batch

                y_pred = model(x)
                y_preds.append(y_pred.cpu())

        y_preds = torch.cat(y_preds)

        if output_type == "one_hot":
            dataset_predictions = to_onehot(torch.argmax(y_preds, dim=1),
                                            num_classes=y_preds.size()[1])
            # to_oneshot returns tensor with uint8 type
            dataset_predictions = dataset_predictions.float()
        elif output_type == "softmax":
            dataset_predictions = F.softmax(y_preds, dim=1)
        elif output_type == "logits":
            dataset_predictions = y_preds
        else:
            self._logger.error(
                    "Model output type must be one of {one_hot, softmax, "
                    "logits}")
            raise ValueError()

        return dataset_predictions

    def run(self):
        raise NotImplementedError("Attacks must implement run method!")
