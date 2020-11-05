from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_trainer(gpus=0, training_epochs=10, early_stop_tolerance=3,
                evaluation_frequency=2, save_loc="./cache", debug=False,
                iteration=None, deterministic=True, validation=True,
                precision=32, accuracy=False):
    # Prepare callbacks
    callbacks = None
    checkpoint_cb = False
    if validation and not debug:
        monitor = "val_acc" if accuracy else "val_f1"
        callbacks = [EarlyStopping(monitor=monitor, verbose=True, mode="max",
                                   patience=early_stop_tolerance)]

        checkpoint_name = "{epoch}-{" + monitor + ":.2f}"
        if iteration is not None:
            checkpoint_name = "iteration={}-".format(iteration) + \
                              checkpoint_name
        checkpoint_cb = ModelCheckpoint(
                filepath=save_loc + "/" + checkpoint_name, mode="max",
                monitor=monitor, verbose=True, save_weights_only=True)

    # Prepare trainer
    trainer = Trainer(default_root_dir=save_loc, gpus=gpus,
                      auto_select_gpus=True if gpus else False,
                      max_epochs=training_epochs,
                      check_val_every_n_epoch=evaluation_frequency,
                      deterministic=deterministic,
                      checkpoint_callback=checkpoint_cb,
                      callbacks=callbacks, fast_dev_run=debug,
                      weights_summary=None, precision=precision)

    return trainer
