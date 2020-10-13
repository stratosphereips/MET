from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_trainer(gpus=0, training_epochs=10, early_stop_tolerance=3,
                evaluation_frequency=2, save_loc="./cache", debug=False,
                iteration=None, deterministic=True, validation=True):
    # Prepare callbacks
    early_stop_cb = EarlyStopping(patience=early_stop_tolerance, verbose=True)

    checkpoint_name = "{epoch}-{val_loss:.2f}-{val_acc:.2f}"
    if iteration is not None:
        checkpoint_name = "iteration={}-".format(iteration) + checkpoint_name
    checkpoint_cb = ModelCheckpoint(
            filepath=save_loc + "/" + checkpoint_name,
            verbose=True, save_weights_only=True)

    # Prepare trainer
    trainer = Trainer(default_root_dir=save_loc, gpus=gpus,
                      auto_select_gpus=True if gpus else False,
                      max_epochs=training_epochs,
                      check_val_every_n_epoch=evaluation_frequency,
                      deterministic=deterministic,
                      early_stop_callback=early_stop_cb,
                      checkpoint_callback=checkpoint_cb,
                      fast_dev_run=debug, weights_summary=None,
                      num_sanity_val_steps=2 if validation else 0,
                      limit_val_batches=1 if validation else 0)

    return trainer
