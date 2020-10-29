import pytorch_lightning as pl
import torch


class MefModule(pl.LightningModule):
    def __init__(self, model, num_classes, optimizer=None, loss=None,
                 lr_scheduler=None):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler
        self.to(next(model.parameters()).device)
        self._f1_macro = pl.metrics.Fbeta(num_classes, average="macro",
                                          compute_on_step=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self._model(x)

        if (y.numel() // len(y)) != 1:
            y = torch.argmax(y, dim=1)

        self._f1_macro(y_hat, y)
        return

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch)
        self.log("val_f1", self._f1_macro, prog_bar=True, on_epoch=True)
        return

    def test_step(self, batch, batch_idx):
        self._shared_step(batch)
        self.log("test_f1", self._f1_macro, on_epoch=True)
        return

    def configure_optimizers(self):
        if self._lr_scheduler is None:
            return self._optimizer
        return [self._optimizer], [self._lr_scheduler]
