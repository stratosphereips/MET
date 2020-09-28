from dataclasses import dataclass
from logging import getLogger

from pytorch_lightning import seed_everything

import mef.attacks.base
from mef.utils.config import Configuration
from mef.utils.logger import set_up_logger


@dataclass
class TestConfig:
    gpus: int = 0
    log_level: str = "info"
    batch_size: int = 64
    evaluation_frequency: int = 2
    early_stop_tolerance: int = 10
    overwrite_models: bool = False
    use_cached_files: bool = False
    seed: int = 0
    debug: bool = False


class Test:
    _config = None
    _logger = None

    def __init__(self, config_file):
        Configuration(config_file)
        self._config = Configuration.get_configuration(TestConfig, "test")

        set_up_logger("Mef", self._config.log_level)
        self._logger = getLogger("Mef")

        seed_everything(self._config.seed)

        # Make the configuration file and logger available framework wide
        mef.attacks.base.Base._logger = self._logger
        mef.attacks.base.Base._test_config = self._config
