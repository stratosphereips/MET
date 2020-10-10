from dataclasses import dataclass
from logging import getLogger
from typing import Optional

from pytorch_lightning import seed_everything

import mef.attacks.base
from mef.utils.config import get_configuration
from mef.utils.logger import set_up_logger


@dataclass
class TestConfig:
    gpus: int
    log_level: str
    seed: Optional[int] = None
    deterministic: bool = True
    debug: bool = False


class Test:
    _config = None
    _logger = None

    def __init__(self, gpus=0, log_level="info", seed=None,
                 deterministic=True, debug=False):
        self._config = get_configuration(TestConfig,
                                         dict(gpus=gpus,
                                              log_level=log_level,
                                              seed=seed,
                                              deterministic=deterministic,
                                              debug=debug))

        set_up_logger("Mef", self._config.log_level)
        self._logger = getLogger("Mef")

        seed_everything(self._config.seed)

        # Make the configuration file and logger available framework wide
        mef.attacks.base.Base._logger = self._logger
        mef.attacks.base.Base._test_config = self._config
