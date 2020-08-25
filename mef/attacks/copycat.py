from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomLabelDataset
from ..attacks.base import Base
from ..utils.pytorch.training import ModelTraining


class CopyCat(Base):
    __default_settings = dict(
        method=["npd", "pd"],
        copycat_model=dict(
            net=dict(
                name="vgg_16"
            )
        ))

    def __init__(self, target_model, copycat_model, test_dataset,
                 problem_domain_dataset, non_problem_domain_dataset):
        super().__init__()

        # Prepare attack's cache folder
        self.__save_loc = Path("cache").joinpath(self._config.test["name"],
                                                 "copycat")
        mkdir_if_missing(self.__save_loc)

        # Attack information
        self._method = '-'.join(self._config.attacks["copycat"]["method"])

        # Datasets
        self._td = test_dataset
        self._pdd = problem_domain_dataset
        self._pdd_sl = None
        self._npdd = non_problem_domain_dataset
        self._npdd_sl = None

        # Models
        self._target_model = target_model
        self._copycat_model = copycat_model

        # Stolen labels
        self._sl_pd = None
        self._sl_npd = None

    def __get_stolen_labels(self, dataset, dataset_type):
        self._logger.info("Getting stolen labels for {} dataset".format(
            ' '.join(dataset_type.split('_'))))

        stolen_labels_file = self.__save_loc.joinpath(dataset_type + ".pth.tar")
        if self._config.test["use_cached_files"]:
            try:
                return torch.load(stolen_labels_file)
            except FileNotFoundError:
                self._logger.warning("{} not found, getting new stolen labels"
                                     .format(stolen_labels_file.name))

        self._target_model.eval()
        loader = DataLoader(dataset, pin_memory=True, batch_size=258,
                            num_workers=1)

        stolen_labels = []
        for batch_idx, (inputs, _) in enumerate(loader, 1):
            if self._config.test["gpu"] is not None:
                inputs = inputs.cuda()
            with torch.no_grad():
                outputs = self._target_model(inputs)
                _, predicted = outputs.max(1)
                stolen_labels.append(predicted.cpu())

                if batch_idx % self._config.test["log_interval"] == 0:
                    self._logger.info("Stolen labels: {}/{} ({:.0f}%)"
                                      .format(batch_idx * len(inputs),
                                              len(loader.dataset),
                                              100. * batch_idx / len(loader)))

        stolen_labels = torch.cat(stolen_labels)
        torch.save(stolen_labels, stolen_labels_file)

        return stolen_labels

    def run(self):

        if "npd" in self._method:
            self._sl_npd = self.__get_stolen_labels(self._npdd,
                                                    "non problem domain")
            self._npdd_sl = CustomLabelDataset(self._npdd, self._sl_npd)
            ModelTraining.train_model(self._copycat_model, training_data=self._npdd_sl,
                                      evaluation_data=self._td)

        if "pd" in self._method:
            self._sl_pd = self.__get_stolen_labels(self._pdd,
                                                   "problem domain")
            self._pdd_sl = CustomLabelDataset(self._pdd, self._sl_pd)
            ModelTraining.train_model(self._copycat_model, training_data=self._pdd_sl,
                                      evaluation_data=self._td)

        return
