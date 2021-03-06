import torch.nn as nn


class Base(nn.Module):
    def __init__(self, num_classes, feature_extraction=False):
        super().__init__()
        self._num_classes = num_classes
        self._feature_extraction = feature_extraction

    def _set_parameter_requires_grad(self, model, feature_extraction) -> None:
        if feature_extraction:
            for param in model.parameters():
                param.requires_grad = False

        return
