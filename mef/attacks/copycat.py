from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomLabelDataset
from ..attacks.base import Base
from ..utils.config import Configuration
from ..utils.pytorch.training import ModelTraining


@dataclass
class CopyCatConfig:
    method: List[str]


class CopyCat(Base):

    def __init__(self, target_model, opd_model, copycat_model, test_dataset,
                 problem_domain_dataset=None, non_problem_domain_dataset=None):
        super().__init__()

        # Get CopyCat's configuration
        self._config = Configuration.get_configuration(CopyCatConfig, "attacks/copycat")

        # Prepare attack's cache folder
        self.__save_loc = Path("cache").joinpath(self._test_config.name, "copycat")
        mkdir_if_missing(self.__save_loc)

        # Attack information
        self._method = '-'.join(self._config.method)

        # Datasets
        self._td = test_dataset
        self._pdd = problem_domain_dataset
        self._pdd_sl = None
        self._npdd = non_problem_domain_dataset
        self._npdd_sl = None

        # Models
        self._target_model = target_model
        self._opd_model = opd_model
        self._copycat_model = copycat_model

        # Stolen labels
        self._sl_pd = None
        self._sl_npd = None

    def _get_stolen_labels(self, dataset, dataset_type):
        self._logger.info(
            "Getting stolen labels for {} dataset".format(' '.join(dataset_type.split('_'))))

        stolen_labels_file = self.__save_loc.joinpath(dataset_type + ".pth.tar")
        if self._test_config.use_cached_files:
            try:
                return torch.load(stolen_labels_file)
            except FileNotFoundError:
                self._logger.warning(
                    "{} not found, getting new stolen labels".format(stolen_labels_file.name))

        self._target_model.eval()
        loader = DataLoader(dataset, pin_memory=True, batch_size=258,
                            num_workers=1)

        stolen_labels = []
        for batch_idx, (inputs, _) in enumerate(loader):
            if self._test_config.gpu is not None:
                inputs = inputs.cuda()
            with torch.no_grad():
                outputs = self._target_model(inputs)
                _, predicted = outputs.max(1)
                stolen_labels.append(predicted.cpu())

                if batch_idx % self._test_config.batch_log_interval == 0:
                    self._logger.debug(
                        "Stolen labels: {}/{} ({:.0f}%)".format(batch_idx * len(inputs),
                                                                len(loader.dataset),
                                                                100. * batch_idx / len(loader)))

        stolen_labels = torch.cat(stolen_labels)
        torch.save(stolen_labels, stolen_labels_file)

        return stolen_labels

    def _training(self, model, training_data, evaluation_data, epochs):
        for epoch in range(epochs):
            self._logger.info("Epoch: {}".format(epoch))
            ModelTraining.train_model(model, training_data, "cross_entropy")

            if epoch % self._test_config.evaluation_frequency:
                eval_accuracy, eval_f1score = ModelTraining.evaluate_model(model, evaluation_data)

                self._logger.info("Evaluation accuracy: {:.3f}\t "
                                  "Evaluation F1-score: {:.3f}".format(eval_accuracy,
                                                                       eval_f1score))

        return

    def run(self):
        self._logger.info("Starting CopyCat attack")
        epochs = self._copycat_model.details.opt.epochs

        if "npd" in self._method:
            self._sl_npd = self._get_stolen_labels(self._npdd, "non_problem_domain")
            self._npdd_sl = CustomLabelDataset(self._npdd, self._sl_npd)
            self._logger.info("Training copycat model with NPD-SL")
            self._training(self._copycat_model, self._npdd_sl, self._td, epochs)

        if "pd" in self._method:
            self._sl_pd = self._get_stolen_labels(self._pdd, "problem_domain")
            self._pdd_sl = CustomLabelDataset(self._pdd, self._sl_pd)
            self._logger.info("Training copycat model with PD-SL")
            self._training(self._copycat_model, self._pdd_sl, self._td, epochs)

        return
