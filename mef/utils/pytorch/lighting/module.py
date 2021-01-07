from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

from mef.utils.pytorch.functional import apply_softmax_or_sigmoid, \
    get_class_labels, get_prob_vector


class _MefModel(pl.LightningModule, ABC):
    def __init__(self,
                 model,
                 num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self._val_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self._f1_macro = pl.metrics.F1(self.num_classes, average="macro",
                                       compute_on_step=False)
        self.test_outputs = None

    def _shared_step(self, batch, step_type):
        x, y = batch

        preds = self(x)[0]

        # preds is expected to in shape of [B] for binary and [B, C] for
        # multiclass
        if preds.size()[-1] == 1:
            preds = preds.squeeze()

        # y is expected to be in shape of [B]
        if y.ndim != 1:
            y = get_class_labels(y)
            if y.size()[-1] == 1:
                y = y.squeeze()

        self._val_accuracy(preds, y)
        self._f1_macro(preds, y)
        self.log_dict({"{}_acc".format(step_type): self._val_accuracy,
                       "{}_f1".format(step_type): self._f1_macro},
                      prog_bar=True, on_epoch=True)
        return preds.detach().cpu()

    def validation_step(self,
                        batch,
                        batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self,
                  batch,
                  batch_idx):
        return self._shared_step(batch, "test")

    def test_epoch_end(self, test_step_outputs):
        # Expected shape is [N, C] for multiclass and [N] for binary
        self.test_outputs = torch.cat(test_step_outputs).squeeze(dim=-1)


class TrainableModel(_MefModel):
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

    @staticmethod
    def _output_to_list(output):
        if isinstance(output, tuple):
            # (logits, hidden_layer)
            return list(output)
        else:
            # logits
            return list([output])

    @auto_move_data
    def forward(self, x):
        output = self.model(x)

        return self._output_to_list(output)

    def training_step(self,
                      batch,
                      batch_idx):
        x, y = batch

        # Dataloader adds one more dimension corresponding to batch size,
        # which means the datasets created by generators in Ripper
        # attacks which already have 4-dimensions will be 5-dimensional in
        # the form [1, B, C, H, W]. In case of y it will be 3-dimensional [1,
        # B, L]
        if len(x.size()) == 5 and x.size()[0] == 1:
            x = x.squeeze(dim=0)
        if len(y.size()) == 3 and y.size()[0] == 1:
            y = y.squeeze(dim=0)

        preds = self(x)[0]
        loss = self._loss(preds, y)

        self.log_dict({"train_loss": loss})
        return loss

    def configure_optimizers(self):
        if self._lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self._lr_scheduler]


class VictimModel(_MefModel):
    def __init__(self,
                 model,
                 num_classes,
                 output_type="prob_dist"):
        super().__init__(model, num_classes)

        if output_type.lower() not in ["one_hot", "prob_dist", "raw",
                                       "labels", "sigmoid/softmax"]:
            raise ValueError("VictimModel output type must be one of {"
                             "one_hot, prob_dist, raw, labels}")

        self._output_type = output_type.lower()

    def _transform_output(self,
                          output):
        if self._output_type == "one_hot":
            y_hats = F.one_hot(torch.argmax(output, dim=-1),
                               num_classes=self.num_classes)
            # to_oneshot returns tensor with uint8 type
            y_hats = y_hats.float()
        elif self._output_type == "sigmoid/softmax":
            y_hats = apply_softmax_or_sigmoid(output)
        elif self._output_type == "labels":
            y_hats = get_class_labels(output)
        else:
            y_hats = output

        return y_hats

    @auto_move_data
    def forward(self, x, inference=True):
        y_hats = self.model(x)

        # Model output must always be 2-dimensional
        if y_hats.ndim == 1:
            y_hats = y_hats.unsqueeze(dim=-1)

        return [self._transform_output(y_hats)]