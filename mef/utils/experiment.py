from pathlib import Path
from typing import Callable, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.lighting.module import TrainableModel
from mef.utils.pytorch.lighting.trainer import get_trainer


def train_victim_model(
    victim_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Callable,
    train_set: Dataset,
    num_classes: int,
    training_epochs: int,
    batch_size: int,
    num_workers: int,
    optimizer_args: Dict[str, Any] = None,
    val_set: Optional[Dataset] = None,
    test_set: Optional[Dataset] = None,
    evaluation_frequency: int = 1,
    patience: int = 100,
    accuracy: bool = False,
    lr_scheduler: Optional[Callable] = None,
    lr_scheduler_args: Dict[str, Any] = None,
    save_loc: str = "./cache/",
    gpu: int = None,
    deterministic: bool = True,
    debug: bool = False,
    precision=32,
) -> None:
    trainer = get_trainer(
        Path(save_loc),
        None,
        training_epochs,
        gpu,
        val_set is not None,
        evaluation_frequency,
        patience,
        accuracy,
        debug,
        deterministic,
        precision,
        logger=True,
    )
    victim_save_dir = Path(save_loc)
    mkdir_if_missing(victim_save_dir)

    try:
        saved_model = torch.load(
            victim_save_dir.joinpath("final_victim_model-state_dict.pt"),
            map_location=torch.device("cpu"),
        )
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
        victim = TrainableModel(
            victim_model,
            num_classes,
            optimizer,
            loss,
            lr_scheduler,
            optimizer_args,
            lr_scheduler_args,
        )
        if gpu is not None:
            victim.cuda(gpu)
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")

        train_dataloader = DataLoader(
            dataset=train_set,
            pin_memory=True if gpu is not None else False,
            num_workers=num_workers,
            shuffle=True,
            batch_size=batch_size,
        )
        val_dataloader = None
        if val_set is not None:
            val_dataloader = DataLoader(
                dataset=val_set,
                pin_memory=True if gpu is not None else False,
                num_workers=num_workers,
                batch_size=batch_size,
            )

        victim = TrainableModel(
            victim_model,
            num_classes,
            optimizer,
            loss,
            lr_scheduler,
            optimizer_args,
            lr_scheduler_args,
        )

        if gpu is not None:
            victim.cuda(gpu)

        trainer.fit(victim, train_dataloader, val_dataloader)

        if trainer.checkpoint_callback.best_model_path is not None:
            checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
            victim.state_dict(checkpoint["state_dict"])
            if gpu is not None:
                victim.cuda(gpu)

        torch.save(
            dict(state_dict=victim.model.state_dict()),
            victim_save_dir.joinpath("final_victim_model-state_dict.pt"),
        )

    if test_set is not None:
        test_dataloader = DataLoader(
            dataset=test_set,
            pin_memory=True if gpu is not None else False,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        trainer.test(victim, test_dataloader)

    return
