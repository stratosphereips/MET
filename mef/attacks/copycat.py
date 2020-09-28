import torch
import torch.nn.functional as F

from mef.utils.pytorch.datasets import CustomLabelDataset, split_data
from .base import Base


class CopyCat(Base):

    def __init__(self, victim_model, substitute_model, test_set, thief_dataset,
                 save_loc="./cache/copycat"):
        super().__init__(save_loc)

        # Datasets
        self._test_set = test_set
        self._thief_dataset = thief_dataset

        # Models
        self._victim_model = victim_model
        self._substitute_model = substitute_model

        # Optimizer, loss_functions
        self._optimizer = torch.optim.SGD(self._substitute_model.parameters(),
                                          lr=0.01, momentum=0.8)
        self._loss = F.cross_entropy

        if self._test_config.gpus:
            self._victim_model.cuda()
            self._substitute_model.cuda()

    def run(self):
        self._logger.info("########### Starting CopyCat attack ###########")

        self._logger.info("Getting stolen labels")
        self._logger.info("Dataset size: {}".format(len(self._thief_dataset)))
        stolen_labels = self._get_predictions(self._victim_model,
                                              self._thief_dataset)
        stolen_labels = torch.argmax(stolen_labels, dim=1)

        synthetic_dataset = CustomLabelDataset(self._thief_dataset,
                                               stolen_labels)

        self._logger.info("Training substitute model with synthetic dataset")
        train_set, val_set = split_data(synthetic_dataset,
                                        self._test_config.val_set_size)
        self._train_model(self._substitute_model, self._optimizer, self._loss,
                          train_set, val_set)

        self._logger.info("Getting substitute model metrics on test set")
        sub_test_acc, sub_test_loss = self._test_model(self._substitute_model,
                                                       self._loss,
                                                       self._test_set)
        self._logger.info(
                "Substitute model Accuracy: {:.1f}% Loss: {:.3f}".format(
                        sub_test_acc, sub_test_loss))

        self._logger.info("Getting final metrics")
        vict_test_labels = self._get_predictions(self._victim_model,
                                                 self._test_set)
        vict_test_labels = torch.argmax(vict_test_labels, dim=1).numpy()
        self._get_attack_metric(self._substitute_model, self._test_set,
                                vict_test_labels)

        final_model_loc = self._save_loc + "/final_substitute_model.pt"
        self._logger.info(
                "Saving final substitute model state dictionary to: {}".format(
                        final_model_loc))
        torch.save(dict(state_dict=self._substitute_model.state_dict),
                   final_model_loc)

        return
