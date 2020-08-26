import torch.nn as nn


class Base(nn.Module):

    def __init__(self, input_dimensions, num_classes, model_details):
        super().__init__()
        self.input_dimensions = input_dimensions
        self.details = model_details
        self.num_classes = num_classes
