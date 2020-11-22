from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from mef.utils.pytorch.datasets import MefDataset
from mef.utils.pytorch.lighting.module import MefModel
from mef.utils.pytorch.lighting.trainer import get_trainer
from mef.utils.settings import BaseSettings


def train_victim_model(victim_model: Module,
                       optimizer: torch.optim.Optimizer,
                       loss: F,
                       train_set: Dataset,
                       num_classes: int,
                       training_epochs: int,
                       batch_size: int,
                       val_set: Dataset = None,
                       evaluation_frequency: int = 1,
                       patience: int = 100,
                       accuracy: bool = False,
                       lr_scheduler: lr_scheduler = None,
                       save_loc: str = "./cache/",
                       gpus: int = 0,
                       deterministic: bool = True,
                       debug: bool = False,
                       precision=32):
    try:
        saved_model = torch.load(Path(save_loc).joinpath("victim",
                                                         "final_victim_model.pt"))
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        # Prepare secret model
        print("Training victim model")

        dataset = MefDataset(BaseSettings(batch_size=batch_size), train_set,
                             val_set)
        train_dataloader = dataset.train_dataloader()

        val_dataloader = None
        if dataset.val_set is not None:
            val_dataloader = dataset.val_dataloader()

        mef_model = MefModel(victim_model, num_classes, optimizer, loss,
                             lr_scheduler)
        trainer = get_trainer(Path(save_loc).joinpath("victim"), None,
                              training_epochs, gpus, dataset.val_set is not
                              None, evaluation_frequency, patience, accuracy,
                              debug, deterministic, precision)
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   Path(save_loc).joinpath("victim", "final_victim_model.pt"))

        return
