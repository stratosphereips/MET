from collections import namedtuple
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mef.utils.pytorch.models.vision.base import Base


class SimpleNet(Base):
    """https://paperswithcode.com/paper/lets-keep-it-simple-using-simple"""

    def __init__(self, dims: Tuple[int, int, int], num_classes: int, return_hidden: bool = False):
        super().__init__(num_classes)

        self._return_hidden = return_hidden
        self._layers = self._make_layers(dims[0])

        self._pool = lambda a: F.max_pool2d(a, kernel_size=a.size()[2:])
        self._drop = nn.Dropout(p=0.1)

        self._fc_final = nn.Linear(256, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self._layers(x)

        # Global Max Pooling
        x = self._pool(x)
        x = self._drop(x)

        hidden = x.view(x.size(0), -1)
        logits = self._fc_final(hidden)

        if self._return_hidden:
            return logits, hidden

        return logits

    def _make_layers(self, input_channels: int) -> nn.Sequential:

        ConvBlock = namedtuple(
            "ConvBlock",
            ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        )
        MaxPoolLayer = namedtuple("MaxPoolLayer", ["kernel_size", "stride"])

        layers = [
            ConvBlock(input_channels, 64, 3, 1, 1),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 256, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(512, 2048, 1, 1, 0),
            ConvBlock(2048, 256, 1, 1, 0),
            MaxPoolLayer(2, 2),
            ConvBlock(256, 256, 3, 1, 1),
        ]

        model = []
        for layer in layers:
            if isinstance(layer, ConvBlock):
                model.extend(
                    [
                        nn.Conv2d(
                            layer.in_channels,
                            layer.out_channels,
                            layer.kernel_size,
                            layer.stride,
                            layer.padding,
                        ),
                        nn.BatchNorm2d(layer.out_channels, momentum=0.05),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                model.extend(
                    [nn.MaxPool2d(layer.kernel_size, layer.stride), nn.Dropout2d(p=0.1)]
                )

        model = nn.Sequential(*model)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        model.apply(init_weights)
        return model
