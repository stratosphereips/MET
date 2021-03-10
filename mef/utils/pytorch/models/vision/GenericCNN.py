from typing import Tuple, Union

import torch
import torch.nn as nn

from mef.utils.pytorch.blocks import ConvBlock
from mef.utils.pytorch.models.vision.base import Base


class GenericCNN(Base):
    def __init__(
        self,
        dims,
        num_classes,
        conv_out_channels=(32, 64, 128),
        fc_layers=(),
        convs_in_block=2,
        dropout_keep_prob=0.1,
        return_hidden=False,
    ):
        super().__init__(num_classes)

        self._return_hidden = return_hidden
        self._dims = dims
        self._conv_out_channels = conv_out_channels
        self._fc_layers = fc_layers
        self._convs_in_block = convs_in_block
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
        conv_out_channels = [self._dims[0]] + list(self._conv_out_channels)
        convs = []
        for idx, out_channels in enumerate(conv_out_channels[1:], start=1):
            for block_idx in range(self._convs_in_block):
                if block_idx != 0:
                    in_channels = convs[-1].conv.out_channels
                else:
                    in_channels = conv_out_channels[idx - 1]

                convs.append(
                    ConvBlock(in_channels, out_channels, kernel_size=(3, 3), padding=1)
                )

            convs.append(nn.MaxPool2d((2, 2), stride=2))
            convs.append(nn.Dropout(p=0.1))

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