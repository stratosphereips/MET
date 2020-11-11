import torch.nn as nn
import torchvision

from mef.utils.pytorch.models.vision.base import Base


class AlexNet(Base):

    def __init__(self, num_classes, feature_extraction=False):
        super().__init__(num_classes, feature_extraction)

        # Load convolutional part of resnet
        self._alexnet = torchvision.models.alexnet(pretrained=True)
        self._set_parameter_requires_grad(self._alexnet,
                                          self._feature_extraction)

        if num_classes != 1000:
            in_features = self._alexnet.classifier[6].in_features
            self._alexnet.classifier[6] = nn.Linear(in_features=in_features,
                                                    out_features=num_classes)

    def forward(self, x, return_all_layers=False):
        modulelist = list(self._alexnet.features.modules())
        for layer in modulelist[:-1]:
            x = layer(x)
        hidden = x
        logits = modulelist[-1](x)

        if return_all_layers:
            return logits, hidden

        return logits
