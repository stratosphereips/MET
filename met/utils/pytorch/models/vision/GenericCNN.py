from typing import Tuple, Union

import torch
import torch.nn as nn

from ...blocks import ConvBlock, MaxPoolLayer
from .base import Base


class GenericCNN(Base):
    def __init__(
        self,
        dims,
        num_classes,
        conv_blocks=(
            ConvBlock(3, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(32, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(64, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            MaxPoolLayer(2, 2),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            MaxPoolLayer(2, 2),
        ),
        fc_layers=(),
        dropout_keep_prob=0.5,
        return_hidden=False,
    ):
        super().__init__(num_classes)

        self._return_hidden = return_hidden
        self._dims = dims
        self._conv_blocks = conv_blocks
        self._fc_layers = fc_layers
        self._dropout_keep_prob = dropout_keep_prob

        self._convs = self._build_convs()

        test_input = torch.zeros(1, dims[0], dims[1], dims[2])
        self.eval()
        test_out = self._convs(test_input)
        num_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self._fcs, self._fc_final = self._build_fcs(num_features)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = self._convs(x)
        hidden = hidden.view(hidden.size(0), -1)

        if len(self._fc_layers) != 0:
            hidden = self._fcs(hidden)

        logits = self._fc_final(hidden)

        if self._return_hidden:
            return logits, hidden

        return logits

    def _build_convs(self):
        convs = []
        for layer in self._conv_blocks:
            if isinstance(layer, ConvBlock):
                convs.extend(
                    [
                        nn.Conv2d(
                            layer.in_channels,
                            layer.out_channels,
                            layer.kernel_size,
                            layer.stride,
                            layer.padding,
                        ),
                        nn.BatchNorm2d(layer.out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                convs.extend(
                    [
                        nn.MaxPool2d(layer.kernel_size, layer.stride, ceil_mode=True),
                        nn.Dropout2d(p=self._dropout_keep_prob),
                    ]
                )

        convs = nn.Sequential(*convs)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        convs.apply(init_weights)

        return convs

    def _build_fcs(self, num_features):
        fcs = []
        for idx, num_neurons in enumerate(self._fc_layers):
            if len(fcs) != 0:
                fcs.append(nn.Linear(fcs[-1].out_channels, num_neurons))
            else:
                fcs.append(nn.Linear(num_features, num_neurons))

        fc_final = nn.Linear(num_features, self._num_classes)
        if len(self._fc_layers) != 0:
            fc_final = nn.Linear(self._fc_layers[-1], self._num_classes)

        return nn.Sequential(*fcs), fc_final
