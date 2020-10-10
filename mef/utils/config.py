# import yaml
from dacite import from_dict


#
#
# class Configuration:
#     config_file = None
#     configs = None
#
#     def __init__(self, config_file):
#         Configuration.config_file = config_file
#         try:
#             with open(self.config_file) as f:
#                 Configuration.configs = yaml.full_load(f)
#         except FileNotFoundError:
#             print("Configuration file {} was not found.".format(
#                 self.config_file))
#
#     @staticmethod
#     def _find_config(configs, keys):
#         for key in configs.keys():
#             if key == keys[0]:
#                 if isinstance(configs[key], dict):
#                     if len(keys) == 1:
#                         return configs[key]
#                     else:
#                         return Configuration._find_config(configs[key],
#                                                           keys[1:])
#
#         return configs
#
#     @classmethod
#     def get_configuration(cls, data_class, dict_):
#         return from_dict(data_class,
#                          cls._find_config(cls.configs, config_type.split(
#                          '/')))

def get_configuration(data_class, dict_):
    return from_dict(data_class, dict_)
