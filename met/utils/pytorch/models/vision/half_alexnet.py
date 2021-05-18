from typing import Tuple

import torch
import torch.nn as nn

from .base import Base


class HalfAlexNet(Base):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 96, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class HalfAlexNetSmall(nn.Module):
    def __init__(self, dims: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(dims[0], 24, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(24, eps=0.001),
            nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=0.001),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=0.001),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64, eps=0.001),
        )

        test_input = torch.zeros(1, dims[0], dims[1], dims[2])
        self.eval()
        test_out = self.features(test_input)
        num_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(256, eps=0.001),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(128, eps=0.001),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
