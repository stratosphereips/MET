import pytorch_lightning as pl
import torch.nn as nn
import torchvision
from pytorch_lightning.core.decorators import auto_move_data

RESNET_TYPES = {"resnet_18": torchvision.models.resnet18,
                "resnet_34": torchvision.models.resnet34,
                "resnet_50": torchvision.models.resnet50,
                "resnet_101": torchvision.models.resnet101,
                "resnet_152": torchvision.models.resnet152}


class ResNet(pl.LightningModule):
    """
    Vgg model architecture using pytorch pretrained models with modifiable
    input size.
    """

    def __init__(self, resnet_type, num_classes, feature_extraction=False):
        super().__init__()

        if resnet_type not in RESNET_TYPES:
            raise ValueError("Unknown resnet_type '{}'".format(resnet_type))

        resnet_loader = RESNET_TYPES[resnet_type]
        self._features = resnet_loader(pretrained=True)
        self._set_parameter_requires_grad(self._features, feature_extraction)

        num_ftrs = self._features.fc.in_features
        self._features.fc = nn.Linear(num_ftrs, num_classes)

    @auto_move_data
    def forward(self, x):
        x = self._features(x)
        return x

    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
