from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.lighting.module import TrainableModel
from mef.utils.pytorch.lighting.trainer import get_trainer


def train_victim_model(victim_model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss: Callable,
                       train_set: Dataset,
                       num_classes: int,
                       training_epochs: int,
                       batch_size: int,
                       num_workers: int,
                       val_set: Optional[Dataset] = None,
                       test_set: Optional[Dataset] = None,
                       evaluation_frequency: int = 1,
                       patience: int = 100,
                       accuracy: bool = False,
                       lr_scheduler: Optional[Callable] = None,
                       save_loc: str = "./cache/",
                       gpus: int = 0,
                       deterministic: bool = True,
                       debug: bool = False,
                       precision=32) -> None:
    trainer, checkpoint_cb = get_trainer(Path(save_loc).joinpath("victim"),
                                         None, training_epochs, gpus,
                                         val_set is not None,
                                         evaluation_frequency, patience,
                                         accuracy, debug, deterministic,
                                         precision, logger=True)
    victim_save_dir = Path(save_loc).joinpath("victim")
    mkdir_if_missing(victim_save_dir)

    try:
        saved_model = torch.load(
                victim_save_dir.joinpath("final_victim_model-state_dict.pt"))
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
        victim = TrainableModel(victim_model, num_classes, optimizer, loss,
                                lr_scheduler)
        if gpus:
            victim.cuda()
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")

        train_dataloader = DataLoader(dataset=train_set, pin_memory=gpus != 0,
                                      num_workers=num_workers, shuffle=True,
                                      batch_size=batch_size)
        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(dataset=val_set, pin_memory=gpus != 0,
                                        num_workers=num_workers,
                                        batch_size=batch_size)

        victim = TrainableModel(victim_model, num_classes, optimizer, loss,
                                lr_scheduler)
        if gpus:
            victim.cuda()

        trainer.fit(victim, train_dataloader, val_dataloader)

        if not isinstance(checkpoint_cb, bool):
            # Load state dictionary of the best model from checkpoint
            checkpoint = torch.load(checkpoint_cb.best_model_path)
            victim.load_state_dict(checkpoint["state_dict"])

        torch.save(dict(state_dict=victim.model.state_dict()),
                   victim_save_dir.joinpath(
                       "final_victim_model-state_dict.pt"))

    if test_set is not None:
        test_dataloader = DataLoader(dataset=test_set, pin_memory=gpus != 0,
                                     num_workers=num_workers,
                                     batch_size=batch_size)
        trainer.test(victim, test_dataloader)

    return
