import torch.nn as nn
import torchvision
from pytorch_lightning.core.decorators import auto_move_data

from mef.models.base import Base

RESNET_TYPES = {"resnet_18": torchvision.models.resnet18,
                "resnet_34": torchvision.models.resnet34,
                "resnet_50": torchvision.models.resnet50,
                "resnet_101": torchvision.models.resnet101,
                "resnet_152": torchvision.models.resnet152}


class ResNet(Base):

    def __init__(self, resnet_type, num_classes, feature_extraction=False):
        super().__init__(num_classes, feature_extraction)

        if resnet_type not in RESNET_TYPES:
            raise ValueError("Unknown resnet_type '{}'".format(resnet_type))

        resnet_loader = RESNET_TYPES[resnet_type]
        self._resnet = resnet_loader(pretrained=True)
        self._set_parameter_requires_grad(self._resnet,
                                          self._feature_extraction)

        in_features = self._resnet.fc.in_features
        self._resnet.fc = nn.Linear(in_features=in_features,
                                    out_features=num_classes)

    @auto_move_data
    def forward(self, x, return_all_layers=False):
        modulelist = list(self._resnet.features.modules())
        for layer in modulelist[:-1]:
            x = layer(x)
        hidden = x
        logits = modulelist[-1](x)

        if return_all_layers:
            return logits, hidden

        return logits
