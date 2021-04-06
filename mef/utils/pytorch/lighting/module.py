from abc import ABC, abstractmethod
from pathlib import Path
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

    def cuda(self) -> None:
        self.to("cuda")
        self._generator.to(self.device)

    @auto_move_data
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._generator(z)


class _MetModel(pl.LightningModule, ABC):
    def __init__(self, model: torch.nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self._val_accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self._f1_macro = torchmetrics.F1(
            self.num_classes, average="macro", compute_on_step=False
        )
        self.test_labels = None

    def cuda(self) -> None:
        self.to("cuda")
        self.model.to(self.device)

    def reset_metrics(self):
        self._val_accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self._f1_macro = torchmetrics.F1(
            self.num_classes, average="macro", compute_on_step=False
        )

    @abstractmethod
    def save(self, save_loc: Path):
        pass

    @abstractmethod
    def _shared_step_output(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], step_type: str
    ) -> torch.Tensor:
        x, y = batch

        preds = self._shared_step_output(x)
        preds = get_class_labels(preds)
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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        # Expected shape is [N, C] for multiclass and [N] for binary
        self.test_labels = get_class_labels(
            torch.cat(test_step_outputs).squeeze(dim=-1)
        ).numpy()
        return


class TrainableModel(_MetModel):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        loss: Callable,
        lr_scheduler: Optional[Callable] = None,
        batch_accuracy: bool = False,
    ):
        super().__init__(model, num_classes)
        self.optimizer = optimizer
        self._loss = loss
        self._lr_scheduler = lr_scheduler
        self._batch_accuracy = batch_accuracy

    def save(self, save_loc: Path):
        torch.save(dict(state_dict=self.model.state_dict()), save_loc)
        return 

    @staticmethod
    def _output_to_list(output: torch.Tensor) -> List[torch.Tensor]:
        if isinstance(output, tuple):
            # (logits, hidden_layer)
            return list(output)
        else:
            # logits
            return list([output])

    @auto_move_data
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        output = self.model(x)

        return self._output_to_list(output)

    def _shared_step_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return torch.sigmoid(self(x)[0])
        return torch.softmax(self(x)[0], dim=-1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, targets = batch

        # Dataloader adds one more dimension corresponding to batch size,
        # which means the datasets created by generators in Ripper
        # attacks which already have 4-dimensions will be 5-dimensional in
        # the form [1, B, C, H, W]. In case of y it will be 3-dimensional [1,
        # B, L]
        if len(x.size()) == 5 and x.size()[0] == 1:
            x = x.squeeze(dim=0)
        if len(targets.size()) == 3 and targets.size()[0] == 1:
            targets = targets.squeeze(dim=0)

        preds = self(x)[0]
        loss = self._loss(preds, targets)

        if self._batch_accuracy:
            batch_acc = preds.max(1)[1].eq(targets.max(1)[1])
            batch_acc = batch_acc.float().mean().detach().cpu()
            self.log("batch_acc", batch_acc, prog_bar=True)

        return loss

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Callable]]
    ]:
        if self._lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self._lr_scheduler]


class VictimModel(_MetModel):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        output_type: str,
    ):
        super().__init__(model, num_classes)

        self.output_type = output_type.lower()
        if self.output_type not in [
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

    def save(self, save_loc: Path):
        pass

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
