from copy import deepcopy

import yaml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class Configuration(metaclass=Singleton):
    """
    Singleton class used to store configuration for test, attacks nad dataset.
    """
    test = None
    attacks = None

    def __init__(self, config_file):
        try:
            with open(config_file) as f:
                configs = yaml.full_load(f)
                self.test = configs["test"]
                self.attacks = configs["attacks"]
        except FileNotFoundError:
            print("Configuration file config.yaml was not found.")

    @classmethod
    def apply_default_settings(cls, original_config, default_config):
        configx = deepcopy(original_config)
        for key in default_config.keys():
            if key in configx:
                if isinstance(configx[key], dict):
                    assert isinstance(default_config[key], dict)
                    if configx[key] != default_config[key]:
                        sub_config = cls.apply_default_settings(configx[key],
                                                                default_config[
                                                                    key])
                        configx[key] = sub_config
            else:
                configx[key] = default_config[key]
        return configx

    @classmethod
    def apply_overwrites(cls, original_config, overwrites):
        configx = deepcopy(original_config)
        for key in overwrites.keys():
            if key in configx:
                if isinstance(configx[key], dict):
                    assert isinstance(overwrites[key], dict)
                    if configx[key] != overwrites[key]:
                        sub_config = cls.apply_overwrites(configx[key],
                                                          overwrites[key])
                        configx[key] = sub_config
                else:
                    configx[key] = overwrites[key]

        return configx
