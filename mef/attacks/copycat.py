import torch
import torch.nn.functional as F

from mef.utils.pytorch.datasets import CustomLabelDataset, split_dataset
from .base import Base


class CopyCat(Base):

    def __init__(self, victim_model, substitute_model, num_classes,
                 training_epochs=1000, early_stop_tolerance=10,
                 evaluation_frequency=2, val_size=0.2, batch_size=64,
                 save_loc="./cache/copycat", gpus=0, seed=None,
                 deterministic=True, debug=False, precision=32,
                 accuracy=False):
        optimizer = torch.optim.SGD(substitute_model.parameters(), lr=0.01,
                                    momentum=0.8)
        loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, optimizer,
                         loss, training_epochs=training_epochs,
                         early_stop_tolerance=early_stop_tolerance,
                         evaluation_frequency=evaluation_frequency,
                         val_size=val_size, batch_size=batch_size,
                         num_classes=num_classes, save_loc=save_loc,
                         gpus=gpus, seed=seed, deterministic=deterministic,
                         debug=debug, precision=precision, accuracy=accuracy)

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)

        self._logger.info("########### Starting CopyCat attack ###########")
        self._logger.info(
                "CopyCat's attack budget: {}".format(len(self._thief_dataset)))

        # Get stolen labels from victim model
        self._logger.info("Getting stolen labels")
        stolen_labels = self._get_predictions(self._victim_model,
                                              self._thief_dataset)
        stolen_labels = torch.argmax(stolen_labels, dim=1)

        synthetic_dataset = CustomLabelDataset(self._thief_dataset,
                                               stolen_labels)
        train_set, val_set = split_dataset(synthetic_dataset, self._val_size)

        self._logger.info("Training substitute model with synthetic dataset")
        self._train_model(self._substitute_model, self._optimizer, train_set,
                          val_set)

        self._get_test_set_metrics()
        self._get_aggreement_score()
        self._save_final_subsitute()

        return
