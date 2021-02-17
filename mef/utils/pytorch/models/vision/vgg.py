import torch
import torch.nn as nn
import torchvision

from mef.utils.pytorch.models.vision.base import Base

VGG_TYPES = {
    "vgg_11": torchvision.models.vgg11,
    "vgg_11_bn": torchvision.models.vgg11_bn,
    "vgg_13": torchvision.models.vgg13,
    "vgg_13_bn": torchvision.models.vgg13_bn,
    "vgg_16": torchvision.models.vgg16,
    "vgg_16_bn": torchvision.models.vgg16_bn,
    "vgg_19_bn": torchvision.models.vgg19_bn,
    "vgg_19": torchvision.models.vgg19,
}


class Vgg(Base):
    def __init__(
        self, vgg_type: str, num_classes: int, feature_extraction: bool = False
    ):
        super().__init__(num_classes, feature_extraction)

        if vgg_type not in VGG_TYPES:
            raise ValueError("Unknown vgg_type '{}'".format(vgg_type))

        # Load convolutional part of vgg
        vgg_loader = VGG_TYPES[vgg_type]
        self._vgg = vgg_loader(pretrained=True)
        self._set_parameter_requires_grad(self._vgg, self._feature_extraction)
        self._vgg.classifier[6].requires_grad_()

        if num_classes != 1000:
            in_features = self._vgg.classifier[6].in_features
            self._vgg.classifier[6] = nn.Linear(
                in_features=in_features, out_features=num_classes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._vgg(x)
