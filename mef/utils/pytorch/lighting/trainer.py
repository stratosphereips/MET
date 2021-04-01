from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mef.utils.settings import BaseSettings, TrainerSettings


def _prepare_callbacks(
    validation: bool,
    patience: int,
    save_loc: Path,
    iteration: int,
    debug: bool,
    accuracy: bool,
) -> Tuple[Optional[List[EarlyStopping]], Union[ModelCheckpoint, bool]]:
    callbacks = None
    checkpoint_cb = False
    if validation and not debug:
        monitor = "val_acc" if accuracy else "val_f1"

        if patience is not None:
            callbacks = [
                EarlyStopping(monitor=monitor, verbose=True, mode="max", patience=patience)
            ]

        checkpoint_name = "{epoch}-{" + monitor + ":.2f}"
        if iteration is not None:
            checkpoint_name = "iteration={}-".format(iteration) + checkpoint_name

        checkpoint_cb = ModelCheckpoint(
            dirpath=save_loc.joinpath("checkpoints"),
            filename=checkpoint_name,
            mode="max",
            monitor=monitor,
            verbose=True,
            save_weights_only=True,
        )

    return callbacks, checkpoint_cb


def get_trainer(
    save_loc: Path,
    iteration: int,
    training_epochs: int,
    gpu: bool,
    validation: bool,
    evaluation_frequency: int,
    patience: int,
    accuracy: bool,
    debug: bool,
    deterministic: bool,
    precision: int,
    logger: bool,
) -> Tuple[Trainer, Union[ModelCheckpoint, bool]]:
    if evaluation_frequency is None:
        evaluation_frequency = np.inf

    callbacks, checkpoint_cb = _prepare_callbacks(
        validation, patience, save_loc, iteration, debug, accuracy
    )

    # Prepare trainer
    trainer = Trainer(
        default_root_dir=save_loc.__str__(),
        gpus=1 if gpu else None,
        auto_select_gpus=True if gpu else False,
        max_epochs=training_epochs,
        min_epochs=training_epochs,
        check_val_every_n_epoch=evaluation_frequency,
        deterministic=deterministic,
        checkpoint_callback=checkpoint_cb,
        callbacks=callbacks,
        fast_dev_run=debug,
        weights_summary=None,
        precision=precision if gpu else 32,
        logger=logger, 
    )

    return trainer, checkpoint_cb


def get_trainer_with_settings(
    base_settings: BaseSettings,
    trainer_settings: TrainerSettings,
    model_name: str = "",
    iteration: int = None,
    validation: bool = False,
    logger: bool = True,
) -> Tuple[Trainer, Union[ModelCheckpoint, bool]]:
    return get_trainer(
        Path(base_settings.save_loc).joinpath(model_name),
        iteration,
        trainer_settings.training_epochs,
        base_settings.gpu,
        validation,
        trainer_settings.evaluation_frequency,
        trainer_settings.patience,
        trainer_settings.use_accuracy,
        base_settings.debug,
        base_settings.deterministic,
        trainer_settings.precision,
        logger,
    )
