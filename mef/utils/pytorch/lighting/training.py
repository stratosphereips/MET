from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_trainer(gpus=0, training_epochs=10, early_stop_tolerance=3, evaluation_frequency=2,
                save_loc="./cache", debug=False):
    # Prepare callbacks
    early_stop_cb = EarlyStopping(patience=early_stop_tolerance, verbose=True)
    checkpoint_cb = ModelCheckpoint(filepath=save_loc + "/{epoch}-{val_loss:.2f}-{val_acc:.2f}",
                                    verbose=True, save_weights_only=True)
    # Prepare trainer
    trainer = Trainer(default_root_dir=save_loc, gpus=gpus,
                      auto_select_gpus=True if gpus else False, max_epochs=training_epochs,
                      check_val_every_n_epoch=evaluation_frequency, deterministic=True,
                      early_stop_callback=early_stop_cb, checkpoint_callback=checkpoint_cb,
                      fast_dev_run=debug)

    return trainer
