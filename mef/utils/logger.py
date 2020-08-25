import logging
import sys
from pathlib import Path

from ..utils.ios import mkdir_if_missing, delete_file


def set_up_logger(config):
    logger_name = config.test["name"]
    log_level = config.test["log_level"]

    log_file = Path("log").joinpath("log_lvl_" + log_level + ".log")
    mkdir_if_missing(log_file.parent)
    delete_file(log_file)

    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "error":
        level = logging.ERROR
    elif log_level == "critical":
        level = logging.CRITICAL
    else:
        raise ValueError(
            "Logging level should be one of {debug, info, warning, error, "
            "critical}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if log_level == "debug":
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(module)s %(funcName)s %("
            "lineno)d: %(message)s")
    else:
        formatter = logging.Formatter(
            "[%(asctime)s]: %(message)s")

    fh = logging.FileHandler(log_file.as_posix())
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
