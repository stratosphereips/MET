import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data


class MefModel(pl.LightningModule):
    def __init__(self,
                 model,
                 num_classes,
                 optimizer=None,
                 loss=None,
                 lr_scheduler=None,
                 output_type="softmax",
                 return_hidden_layer=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler
        self.to(next(model.parameters()).device)
        self._train_accuracy = pl.metrics.Accuracy()
        self._accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self._f1_macro = pl.metrics.Fbeta(num_classes, average="macro",
                                          compute_on_step=False)
        self._output_type = output_type
        self._return_hidden_layer = return_hidden_layer

    @auto_move_data
    def forward(self, x):
        if self._return_hidden_layer:
            output, hidden = self.model(x)
        else:
            output = self.model(x)

        if self._output_type == "one_hot":
            y_hats = F.one_hot(torch.argmax(output, dim=-1),
                               num_classes=output.size()[-1])
            # to_oneshot returns tensor with uint8 type
            y_hats = y_hats.float()
        elif self._output_type == "softmax":
            y_hats = F.softmax(output, dim=-1)
        elif self._output_type == "labels":
            y_hats = torch.argmax(output, dim=-1)
        else:
            y_hats = output

        if self._return_hidden_layer:
            return [y_hats, hidden]
        else:
            return [y_hats]

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
        output = self.model(x)

        if (y.numel() // len(y)) != 1:
            y = torch.argmax(y, dim=-1)

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
