from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from mef.utils.pytorch.lighting.module import MefModule


def train_victim_model(model, optimizer, loss, train_set, val_set,
                       training_epochs, save_loc, gpus):
    mef_model = MefModule(model, optimizer, loss)

    trainer = get_trainer(gpus, training_epochs, save_loc=save_loc)

    train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                  pin_memory=True, num_workers=4,
                                  batch_size=64)
    val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                num_workers=4, batch_size=64)

    trainer.fit(mef_model, train_dataloader, val_dataloader)

    return


def get_trainer(gpus=0, training_epochs=10, early_stop_tolerance=3,
                evaluation_frequency=2, save_loc="./cache", debug=False,
                iteration=None, deterministic=True):
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
                      fast_dev_run=debug)

    return trainer
