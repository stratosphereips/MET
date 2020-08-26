import torch.nn as nn
import torch.nn.functional as F

from mef.models.base import Base
from mef.utils.pytorch.blocks import LinearBlock, ConvBlock, return_act, \
    return_drop


class Mnet(Base):
    """
    From whitenblackbox paper. This convnet structure cover many LeNet variants.
    """

    def __init__(self, input_dimensions, num_classes, model_details):
        super().__init__(input_dimensions, num_classes, model_details)

        self._act = return_act(self.details.net.act)
        self._ks = self.details.net.ks

        self._conv1 = nn.Conv2d(self.input_dimensions[0], 10, kernel_size=self._ks)
        self._conv2 = nn.Conv2d(10, 20, kernel_size=self._ks)

        conv_iter = []
        for _ in range(self.details.net.n_conv - 2):
            conv_iter.append(ConvBlock(
                20, 20, self._ks, padding=int((self._ks - 1) / 2)))
        self._conv_iter = nn.Sequential(*conv_iter)

        self._pool = lambda a: a
        pool_factor = 1
        if "max" in self.details.net.pool:
            stride = int(self.details.net.pool.split('_')[1])
            self._pool = lambda a: F.max_pool2d(a, stride)
            pool_factor = 2

        self._fc_feat_dim = int((
                                        (
                                                (
                                                        (
                                                                (
                                                                        self.input_dimensions[1]
                                                                        -
                                                                        self._ks + 1) // pool_factor
                                                        ) - self._ks + 1) //
                                                pool_factor
                                        ) ** 2
                                ) * 20)

        self._fc1 = nn.Linear(self._fc_feat_dim, 50)

        self._drop = lambda a: a
        if self.details.net.drop != "none":
            drop = return_drop(self.details.net.drop)
            self._drop = lambda a: drop(a)

        fc_iter = []
        for _ in range(self.details.net.fc - 2):
            fc_iter.append(
                LinearBlock(50, 50, self.details.net.drop))
        self._fc_iter = nn.Sequential(*fc_iter)
        self._fc_final = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = self._act(self._pool(self._conv1(x)))
        x = self._act(self._pool(self._conv2(x)))
        x = self._conv_iter(x)
        x = x.view(-1, self._fc_feat_dim)
        x = self._act(self._fc1(x))
        x = self._drop(x)
        x = self._fc_iter(x)
        x = self._fc_final(x)
        return x
