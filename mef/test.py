from dataclasses import dataclass
from logging import getLogger

import torch
from ignite.utils import manual_seed

import mef.attacks.base
from mef.utils.config import Configuration
from mef.utils.logger import set_up_logger


@dataclass
class TestConfig:
    name: str = "Mef"
    cuda: bool = False
    log_level: str = "info"
    batch_size: int = 64
    evaluation_frequency: int = 2
    early_stop_tolerance: int = 10
    overwrite_models: bool = False
    use_cached_files: bool = False
    seed: int = 0


class Test:
    _config = None
    _logger = None

    def __init__(self, config_file):
        Configuration(config_file)
        self._config = Configuration.get_configuration(TestConfig, "test")
        self._config.cuda = torch.cuda.is_available()

        set_up_logger(self._config.name, self._config.log_level)
        self._logger = getLogger(self._config.name)

        manual_seed(self._config.seed)

        # Make the configuration file and logger available framework wide
        mef.attacks.base.Base._logger = self._logger
        mef.attacks.base.Base._test_config = self._config
