from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

from mef.utils.pytorch.functional import get_labels, get_prob_dist


def _tranfsform_output(output,
                       output_type):
    if output_type == "one_hot":
        y_hats = F.one_hot(torch.argmax(output, dim=-1),
                           num_classes=output.size()[-1])
        # to_oneshot returns tensor with uint8 type
        y_hats = y_hats.float()
    elif output_type == "prob_dist":
        y_hats = get_prob_dist(output)
    elif output_type == "labels":
        y_hats = get_labels(output)
    else:
        y_hats = output

    return y_hats


def _check_float_type(tensor):
    if tensor.dtype.__str__() != "torch.float32":
        return tensor.float()
    return tensor


class MefModel(pl.LightningModule, ABC):
    def __init__(self,
                 model,
                 num_classes):
        super().__init__()
        self.model = model

        self._accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self._f1_macro = pl.metrics.Fbeta(num_classes, average="macro",
                                          compute_on_step=False)

    @abstractmethod
    def _shared_step(self, batch):
        pass

    def validation_step(self,
                        batch,
                        batch_idx):
        self._shared_step(batch)
        self.log_dict({"val_acc": self._accuracy, "val_f1": self._f1_macro},
                      prog_bar=True, on_epoch=True)
        return

    def test_step(self,
                  batch,
                  batch_idx):
        self._shared_step(batch)
        self.log_dict({"test_acc": self._accuracy, "test_f1": self._f1_macro},
                      prog_bar=True, on_epoch=True)
        return


class TrainingModel(MefModel):
    def __init__(self,
                 model,
                 num_classes,
                 optimizer=None,
                 loss=None,
                 lr_scheduler=None):
        super().__init__(model, num_classes)
        self.optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler

    @auto_move_data
    def forward(self, x):
        output = self.model(x)

        if isinstance(output, tuple):
            # [logits, hidden_layer]
            return list(output)
        else:
            # [logits]
            return list([output])

    def training_step(self,
                      batch,
                      batch_idx):
        x, y = batch

        # Dataloader adds one more dimension corresponding to batch size,
        # which means the datasets created by generators which already
        # have 4-dimensions will be 5-dimensional in the form [1, B, C, H, W]
        # In case of y it will be 3-dimensional [1, B, L]
        if len(x.size()) == 5:
            x = x.squeeze()
        if len(y.size()) == 3:
            y = y.squeeze()

        logits = self.model(x)
        loss = self._loss(logits, y)

        self.log("train_loss", loss)
        return loss

    def _shared_step(self, batch):
        x, y = batch
        x = _check_float_type(x)
        y = _check_float_type(y)

        logits = self.model(x)
        output = _tranfsform_output(logits, "prob_dist")

        if len(y.size()) == 1:
            y = torch.round(y)
        else:
            y = torch.argmax(y, dim=-1)

        output = _check_float_type(output)
        self._accuracy(output, y)
        self._f1_macro(output, y)
        return

    def validation_step(self,
                        batch,
                        batch_idx):
        self._shared_step(batch)
        self.log_dict({"val_acc": self._accuracy, "val_f1": self._f1_macro},
                      prog_bar=True, on_epoch=True)
        return

    def test_step(self,
                  batch,
                  batch_idx):
        self._shared_step(batch)
        self.log_dict({"test_acc": self._accuracy, "test_f1": self._f1_macro},
                      prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        if self._lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self._lr_scheduler]


class VictimModel(MefModel):
    def __init__(self,
                 model,
                 num_classes,
                 output_type="prob_dist"):
        super().__init__(model, num_classes)
        self._output_type = output_type

    @auto_move_data
    def forward(self, x, inference=True):
        output = self.model(x)

        y_hats = _tranfsform_output(output, self._output_type)

        # In case the underlying model is not on GPU but on CPU
        if self.device.type == "cuda":
            y_hats = y_hats.cuda()

        return [y_hats]

    def _shared_step(self, batch):
        x, y = batch
        x = _check_float_type(x)
        y = _check_float_type(y)

        output = self.model(x)

        # In case the underlying victim model is not on GPU but on CPU
        if self.device.type == "cuda":
            output = output.cuda()

        if len(y.size()) == 1:
            y = torch.round(y).long()
        else:
            y = torch.argmax(y, dim=-1).long()

        output = _check_float_type(output)
        self._accuracy(output, y)
        self._f1_macro(output, y)
        return
