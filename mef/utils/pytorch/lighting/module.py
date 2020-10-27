import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class MefModule(pl.LightningModule):
    def __init__(self, model, optimizer=None, loss=None, lr_scheduler=None):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler
        self.to(next(model.parameters()).device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def _shared_step(self, batch):
        x, y = batch

        y_hat = self._model(x)

        loss = self._loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

        return

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)

        return

    def configure_optimizers(self):
        if self._lr_scheduler is None:
            return self._optimizer
        return [self._optimizer], [self._lr_scheduler]
