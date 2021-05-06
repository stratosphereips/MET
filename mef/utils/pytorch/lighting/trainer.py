from pathlib import Path
from typing import List, Union

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
) -> Union[List[Union[EarlyStopping, ModelCheckpoint]], None]:
    callbacks = None
    checkpoint_cb = False
    if validation and not debug:
        monitor = "val_acc" if accuracy else "val_f1"

        callbacks = []
        if patience is not None:
            callbacks = [
                EarlyStopping(
                    monitor=monitor, verbose=True, mode="max", patience=patience
                )
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
        callbacks.append(checkpoint_cb)

    return callbacks


def get_trainer(
    save_loc: Path,
    iteration: int,
    training_epochs: int,
    gpu: int,
    validation: bool,
    evaluation_frequency: int,
    patience: int,
    accuracy: bool,
    debug: bool,
    deterministic: bool,
    precision: int,
    logger: bool,
) -> Trainer:
    if evaluation_frequency is None:
        evaluation_frequency = np.inf

    callbacks = _prepare_callbacks(
        validation, patience, save_loc, iteration, debug, accuracy
    )

    # Prepare trainer
    trainer = Trainer(
        default_root_dir=save_loc.__str__(),
        checkpoint_callback=True if callbacks is not None else False,
        gpus=str(gpu),
        max_epochs=training_epochs,
        check_val_every_n_epoch=evaluation_frequency,
        deterministic=deterministic,
        callbacks=callbacks,
        fast_dev_run=debug,
        weights_summary=None,
        precision=precision if gpu is not None else 32,
        logger=logger,
    )

    return trainer


def get_trainer_with_settings(
    base_settings: BaseSettings,
    trainer_settings: TrainerSettings,
    model_name: str = "",
    iteration: int = None,
    validation: bool = False,
    logger: bool = True,
) -> Trainer:
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
