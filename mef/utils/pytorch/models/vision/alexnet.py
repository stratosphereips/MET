import torch
import torch.nn as nn
import torchvision

from mef.utils.pytorch.models.vision.base import Base


class AlexNet(Base):

    def __init__(self, num_classes, pretrained=True, feature_extraction=False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._alexnet(x)


class AlexNetSmall(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        test_input = torch.zeros(1, dims[0], dims[1], dims[2])
        # nn.BatchNorm expects more than 1 value
        self.eval()
        test_out = self.features(test_input)
        num_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(num_features, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
