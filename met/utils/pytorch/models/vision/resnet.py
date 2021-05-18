import torch
import torch.nn as nn
import torchvision

from .base import Base

RESNET_TYPES = {
    "resnet_18": torchvision.models.resnet18,
    "resnet_34": torchvision.models.resnet34,
    "resnet_50": torchvision.models.resnet50,
    "resnet_101": torchvision.models.resnet101,
    "resnet_152": torchvision.models.resnet152,
}


class ResNet(Base):
    def __init__(
        self,
        resnet_type: str,
        num_classes: int,
        feature_extraction: bool = False,
        smaller_resolution: bool = True,
    ):
        super().__init__(num_classes, feature_extraction)

        if resnet_type not in RESNET_TYPES:
            raise ValueError("Unknown resnet_type '{}'".format(resnet_type))

        resnet_loader = RESNET_TYPES[resnet_type]
        self._resnet = resnet_loader(pretrained=True)
        self._set_parameter_requires_grad(self._resnet, self._feature_extraction)
        self._resnet.fc.requires_grad_()

        if smaller_resolution:
            self._resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        if num_classes != 1000:
            in_features = self._resnet.fc.in_features
            self._resnet.fc = nn.Linear(
                in_features=in_features, out_features=num_classes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._resnet(x)
