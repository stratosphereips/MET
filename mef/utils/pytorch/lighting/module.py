from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

from mef.utils.pytorch.functional import get_class_labels


class Generator(pl.LightningModule):
    def __init__(self, generator: torch.nn.Module, latent_dim: int):
        super().__init__()
        self._generator = generator
        self.latent_dim = latent_dim

    @auto_move_data
    def forward(self, z):
        return self._generator(z)


class _MefModel(pl.LightningModule, ABC):
    def __init__(self, model: torch.nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self._val_accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self._f1_macro = torchmetrics.F1(
            self.num_classes, average="macro", compute_on_step=False
        )
        self.test_labels = None

    @abstractmethod
    def _shared_step_output(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], step_type: str
    ) -> torch.Tensor:
        x, y = batch

        preds = self._shared_step_output(x)

        # preds is expected to be int type if containing only labels in muli-class setting
        if self.num_classes > 2 and preds.ndim == 1:
            preds = preds.int()

        # y is expected to be in shape of [B]
        if y.size() != 1:
            y = get_class_labels(y)

        self._val_accuracy(preds, y)
        self._f1_macro(preds, y)
        self.log_dict(
            {f"{step_type}_acc": self._val_accuracy, f"{step_type}_f1": self._f1_macro},
            prog_bar=True,
            on_epoch=True,
        )
        return preds.detach().cpu()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        # Expected shape is [N, C] for multiclass and [N] for binary
        self.test_labels = get_class_labels(
            torch.cat(test_step_outputs).squeeze(dim=-1)
        ).numpy()
        return


class TrainableModel(_MefModel):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        loss: Callable,
        lr_scheduler: Optional[Callable] = None,
    ):
        super().__init__(model, num_classes)
        self.optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler

    @staticmethod
    def _output_to_list(output: torch.Tensor) -> List[torch.Tensor]:
        if isinstance(output, tuple):
            # (logits, hidden_layer)
            return list(output)
        else:
            # logits
            return list([output])

    @auto_move_data
    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor]]:
        output = self.model(x)

        return self._output_to_list(output)

    def _shared_step_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return torch.sigmoid(self(x)[0])
        return torch.softmax(self(x)[0], dim=-1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Callable]]
    ]:
        if self._lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self._lr_scheduler]


class VictimModel(_MefModel):
    def __init__(self, model: torch.nn.Module, num_classes: int, output_type: str):
        super().__init__(model, num_classes)

        if output_type.lower() not in [
            "one_hot",
            "raw",
            "logits",
            "labels",
            "sigmoid",
            "softmax",
        ]:
            raise ValueError(
                "VictimModel output type must be one of {"
                "one_hot, raw, logits, labels, sigmoid,"
                "softmax}"
            )

        self.output_type = output_type.lower()

    def _transform_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.output_type == "one_hot":
            y_hats = F.one_hot(
                torch.argmax(output, dim=-1), num_classes=self.num_classes
            )
            # to_oneshot returns tensor with uint8 type
            y_hats = y_hats.float()
        elif self.output_type == "sigmoid":
            y_hats = torch.sigmoid(output)
        elif self.output_type == "softmax":
            y_hats = torch.softmax(output, dim=-1)
        elif self.output_type == "labels":
            y_hats = get_class_labels(output)
        else:
            y_hats = output

        return y_hats

    @auto_move_data
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y_hats = self.model(x)

        # Model output must always be 2-dimensional
        if y_hats.ndim == 1:
            y_hats = y_hats.unsqueeze(dim=-1)

        return [self._transform_output(y_hats)]

    def _shared_step_output(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)[0]
