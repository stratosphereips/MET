import torch.nn as nn
import torchvision

from mef.models.base import Base


class AlexNet(Base):
    """
    AlexNet model architecture using pytorch pretrained models with modifiable
    input size.
    """

    def __init__(self, num_classes, feature_extraction=False):
        super().__init__(num_classes, feature_extraction)

        # Load convolutional part of resnet
        self._alexnet = torchvision.models.alexnet(pretrained=True)
        self._set_parameter_requires_grad(self._alexnet,
                                          self._feature_extraction)

        in_features = self._alexnet.classifier[6].in_features
        self._classifier[6] = nn.Linear(in_features=in_features,
                                        out_features=num_classes)

    def forward(self, x):
        logits = self._alexnet(x)
        return logits
