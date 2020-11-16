from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mef.utils.settings import BaseSettings, TrainerSettings


def _prepare_callbacks(validation: bool,
                       patience: int,
                       save_loc: Path,
                       iteration: int,
                       debug: bool,
                       accuracy: bool
                       ):
    callbacks = None
    checkpoint_cb = False
    if validation and not debug:
        monitor = "val_acc" if accuracy else "val_f1"
        callbacks = [EarlyStopping(monitor=monitor, verbose=True, mode="max",
                                   patience=patience)]

        checkpoint_name = "{epoch}-{" + monitor + ":.2f}"
        if iteration is not None:
            checkpoint_name = "iteration={}-".format(iteration) + \
                              checkpoint_name
        filepath = save_loc.joinpath(checkpoint_name)
        checkpoint_cb = ModelCheckpoint(filepath=filepath.__str__(),
                                        mode="max", monitor=monitor,
                                        verbose=True, save_weights_only=True)

    return callbacks, checkpoint_cb


def get_trainer(save_loc: Path,
                iteration: int,
                training_epochs: int,
                gpus: int,
                validation: bool,
                evaluation_frequency: int,
                patience: int,
                accuracy: bool,
                debug: bool,
                deterministic: bool,
                precision: int):
    callbacks, checkpoint_cb = _prepare_callbacks(validation, patience,
                                                  save_loc, iteration,
                                                  debug, accuracy)

    # Prepare trainer
    trainer = Trainer(default_root_dir=save_loc.__str__(),
                      gpus=gpus,
                      auto_select_gpus=True if gpus else False,
                      max_epochs=training_epochs,
                      check_val_every_n_epoch=evaluation_frequency,
                      deterministic=deterministic,
                      checkpoint_callback=checkpoint_cb,
                      callbacks=callbacks,
                      fast_dev_run=debug,
                      weights_summary=None,
                      precision=precision)

    return trainer


def get_trainer_with_settings(base_settings: BaseSettings,
                              trainer_settings: TrainerSettings,
                              model_name: str,
                              iteration: int,
                              validation: bool):
    return get_trainer(Path(base_settings.save_loc).joinpath(model_name),
                       iteration, trainer_settings.training_epochs,
                       base_settings.gpus, validation,
                       trainer_settings.evaluation_frequency,
                       trainer_settings.patience, trainer_settings.accuracy,
                       base_settings.debug, base_settings.deterministic,
                       trainer_settings.precision)
