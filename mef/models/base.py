import torch.nn as nn


class Base(nn.Module):

    def __init__(self, sample_dimensions, n_classes, model_config):
        super().__init__()
        self.sample_dimensions = sample_dimensions
        self.config = model_config
        self.n_classes = n_classes
