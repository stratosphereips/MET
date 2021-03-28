import torch
import torch.nn as nn

from mef.utils.pytorch.models.vision.base import Base


class LeNet(Base):
    def __init__(
        self, num_classes, return_hidden=False,
    ):
        super().__init__(num_classes)
        self._return_hidden = return_hidden

        self._convs = [
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]

        self._convs = nn.Sequential(*self._convs)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        self._convs.apply(init_weights)

        self._fcs = [
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        ]

        self._fcs = nn.Sequential(*self._fcs)

    def forward(self, x):
        hidden = self._convs(x)
        hidden = hidden.view(-1, 16 * 5 * 5)
        logits = self._fcs(hidden)

        if self._return_hidden:
            return logits, hidden

        return logits
