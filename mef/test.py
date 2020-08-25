import random
from logging import getLogger

import numpy as np
import torch

import mef.attacks.base
import mef.utils.config
import mef.utils.logger
import mef.utils.pytorch.training


class Test:
    _config = None
    _logger = None

    default_settings = dict(overwrite_models=False,
                            gpu=None,
                            log_level="info",
                            log_interval=100,
                            seed=0)

    def __init__(self, config_file):
        self._config = mef.utils.config.Configuration(config_file)
        mef.utils.logger.set_up_logger(self._config)
        self._logger = getLogger(self._config.test["name"])

        # Make the configuration file and logger available framework wide
        mef.attacks.base.Base._logger = self._logger
        mef.attacks.base.Base._config = self._config
        mef.utils.pytorch.training.ModelTraining._logger = self._logger
        mef.utils.pytorch.training.ModelTraining._config = self._config

    def _set_seed(self):
        self._logger.debug("Setting seed for random generators.")

        torch.manual_seed(self._config.test["seed"])
        torch.cuda.manual_seed(self._config.test["seed"])
        random.seed(self._config.test["seed"])
        np.random.seed(self._config.test["seed"])

        return
