import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import functional as FM


class MefModule(pl.LightningModule):
    def __init__(self, model, optimizer=None, loss=None, lr_scheduler=None,
                 labels=True):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler
        self._labels = labels

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)

        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        return result

    def validation_step(self, batch, batch_idx, labels=True):
        x, y = batch

        if not labels:
            y = torch.argmax(y, dim=1)

        y_hat = self._model(x)

        loss = self._loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx, self._labels)
        result.rename_keys({"val_acc": "test_acc", "val_loss": "test_loss"})
        return result

    def configure_optimizers(self):
        if self._lr_scheduler is None:
            return self._optimizer
        return self._optimizer, self._lr_scheduler
