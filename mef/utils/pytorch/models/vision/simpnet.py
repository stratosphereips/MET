from collections import namedtuple
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mef.utils.pytorch.models.vision.base import Base


class SimpNet(Base):

    def __init__(self,
                 num_classes: int,
                 return_hidden: bool = False):
        super().__init__(num_classes)

        self._return_hidden = return_hidden
        self._layers = self._make_layers()

        self._pool = lambda a: F.max_pool2d(a, kernel_size=a.size()[2:])
        self._drop = nn.Dropout(p=0.2)

        self._fc_final = nn.Linear(432, num_classes)

    def forward(self,
                x: torch.Tensor) -> Union[torch.Tensor,
                                          Tuple[torch.Tensor, torch.Tensor]]:
        x = self._layers(x)

        # Global Max Pooling
        x = self._pool(x)
        x = self._drop(x)

        hidden = x.view(x.size(0), -1)
        logits = self._fc_final(hidden)

        if self._return_hidden:
            return logits, hidden

        return logits

    def _make_layers(self) -> nn.Sequential:

        ConvBlock = namedtuple("ConvBlock", ["in_channels", "out_channels",
                                             "kernel_size", "stride",
                                             "padding"])
        MaxPoolLayer = namedtuple("MaxPoolLayer", ["kernel_size", "stride"])

        layers = [ConvBlock(3, 66, 3, 1, 1), ConvBlock(66, 128, 3, 1, 1),
                  ConvBlock(128, 128, 3, 1, 1), ConvBlock(128, 128, 3, 1, 1),
                  ConvBlock(128, 192, 3, 1, 1), MaxPoolLayer(2, 2),
                  ConvBlock(192, 192, 3, 1, 1), ConvBlock(192, 192, 3, 1, 1),
                  ConvBlock(192, 192, 3, 1, 1), ConvBlock(192, 192, 3, 1, 1),
                  ConvBlock(192, 288, 3, 1, 1), MaxPoolLayer(2, 2),
                  ConvBlock(288, 288, 3, 1, 1), ConvBlock(288, 355, 3, 1, 1),
                  ConvBlock(355, 432, 3, 1, 1)]

        model = []
        for layer in layers:
            if isinstance(layer, ConvBlock):
                model.extend([nn.Conv2d(layer.in_channels, layer.out_channels,
                                        layer.kernel_size, layer.stride,
                                        layer.padding),
                              nn.BatchNorm2d(layer.out_channels,
                                             momentum=0.05),
                              nn.ReLU(inplace=True), nn.Dropout2d(p=0.2)])
            else:
                model.extend([nn.MaxPool2d(layer.kernel_size, layer.stride),
                              nn.Dropout2d(p=0.2)])

        model = nn.Sequential(*model)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data,
                                        gain=nn.init.calculate_gain("relu"))

        model.apply(init_weights)
        return model