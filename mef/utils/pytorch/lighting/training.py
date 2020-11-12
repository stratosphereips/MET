from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_trainer(base_settings,
                trainer_settings,
                model_name: str = "",
                iteration: int = None):
    # Prepare callbacks
    callbacks = None
    checkpoint_cb = False
    if trainer_settings._validation and not base_settings.debug:
        monitor = "val_acc" if trainer_settings.accuracy else "val_f1"
        callbacks = [EarlyStopping(monitor=monitor, verbose=True, mode="max",
                                   patience=trainer_settings.patience)]

        checkpoint_name = "{epoch}-{" + monitor + ":.2f}"
        if iteration is not None:
            checkpoint_name = "iteration={}-".format(iteration) + \
                              checkpoint_name
        filepath = Path(base_settings.save_loc).joinpath(model_name,
                                                         checkpoint_name)

        checkpoint_cb = ModelCheckpoint(filepath=filepath, mode="max",
                                        monitor=monitor, verbose=True,
                                        save_weights_only=True)

    # Prepare trainer
    trainer = Trainer(default_root_dir=base_settings.save_loc + model_name,
                      gpus=base_settings.gpus,
                      auto_select_gpus=True if base_settings.gpus else False,
                      max_epochs=trainer_settings.training_epochs,
                      check_val_every_n_epoch=trainer_settings
                      .evaluation_frequency,
                      deterministic=base_settings.deterministic,
                      checkpoint_callback=checkpoint_cb,
                      callbacks=callbacks,
                      fast_dev_run=base_settings.debug,
                      weights_summary=None,
                      precision=trainer_settings.precision)

    return trainer
