from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from mef.utils.pytorch.models.vision.base import Base


class AlexNet(Base):

    def __init__(self,
                 num_classes: int,
                 pretrained: bool = True,
                 feature_extraction: bool = False):
        super().__init__(num_classes, feature_extraction)

        # Load convolutional part of resnet
        self._alexnet = torchvision.models.alexnet(pretrained=pretrained)
        self._set_parameter_requires_grad(self._alexnet,
                                          self._feature_extraction)
        self._alexnet.classifier[6].requires_grad_()

        if num_classes != 1000:
            in_features = self._alexnet.classifier[6].in_features
            self._alexnet.classifier[6] = nn.Linear(in_features=in_features,
                                                    out_features=num_classes)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self._alexnet(x)


class AlexNetSmall(nn.Module):
    def __init__(self,
                 dims: Tuple[int, int, int],
                 num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(48, eps=0.001),
                nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(128, eps=0.001),
                nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192, eps=0.001),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192, eps=0.001),
                nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(128, eps=0.001)
        )

        test_input = torch.zeros(1, dims[0], dims[1], dims[2])
        self.eval()
        test_out = self.features(test_input)
        num_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.BatchNorm1d(512, eps=0.001),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.BatchNorm1d(256, eps=0.001),
                nn.Linear(256, num_classes),
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
