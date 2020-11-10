import torch.nn as nn
import torchvision
from pytorch_lightning.core.decorators import auto_move_data

from mef.models.vision.base import Base

VGG_TYPES = {"vgg_11": torchvision.models.vgg11,
             "vgg_11_bn": torchvision.models.vgg11_bn,
             "vgg_13": torchvision.models.vgg13,
             "vgg_13_bn": torchvision.models.vgg13_bn,
             "vgg_16": torchvision.models.vgg16,
             "vgg_16_bn": torchvision.models.vgg16_bn,
             "vgg_19_bn": torchvision.models.vgg19_bn,
             "vgg_19": torchvision.models.vgg19}


class Vgg(Base):

    def __init__(self, vgg_type, num_classes, feature_extraction=False):
        super().__init__(num_classes, feature_extraction)

        if vgg_type not in VGG_TYPES:
            raise ValueError("Unknown vgg_type '{}'".format(vgg_type))

        # Load convolutional part of vgg
        vgg_loader = VGG_TYPES[vgg_type]
        self._vgg = vgg_loader(pretrained=True)
        self._set_parameter_requires_grad(self._vgg,
                                          self._feature_extraction)

        if num_classes != 1000:
            in_features = self._vgg.classifier[6].in_features
            self._vgg.classifier[6] = nn.Linear(in_features=in_features,
                                                out_features=num_classes)

    @auto_move_data
    def forward(self, x, return_all_layers=False):
        modulelist = list(self._vgg.features.modules())
        for layer in modulelist[:-1]:
            x = layer(x)
        hidden = x
        logits = modulelist[-1](x)

        if return_all_layers:
            return logits, hidden

        return logits
