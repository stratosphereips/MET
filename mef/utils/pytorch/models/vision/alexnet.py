import torch.nn as nn
import torchvision
from pytorch_lightning.core.decorators import auto_move_data

from mef.utils.pytorch.models.vision.base import Base


class AlexNet(Base):

    def __init__(self, num_classes, feature_extraction=False):
        super().__init__(num_classes, feature_extraction)

        # Load convolutional part of resnet
        self._alexnet = torchvision.models.alexnet(pretrained=True)
        self._set_parameter_requires_grad(self._alexnet,
                                          self._feature_extraction)
        self._alexnet.classifier[6].requires_grad_()

        if num_classes != 1000:
            in_features = self._alexnet.classifier[6].in_features
            self._alexnet.classifier[6] = nn.Linear(in_features=in_features,
                                                    out_features=num_classes)

    @auto_move_data
    def forward(self, x):
        return  self._alexnet(x)
