import torch.nn as nn


class Base(nn.Module):

    def __init__(self, sample_dimensions, n_classes, model_details):
        super().__init__()
        self.sample_dimensions = sample_dimensions
        self.details = model_details
        self.n_classes = n_classes
