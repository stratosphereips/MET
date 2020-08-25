import torch.nn as nn
import torch.nn.functional as F

from mef.models.base import Base
from mef.utils.pytorch.blocks import LinearBlock, ConvBlock, return_act, \
    return_drop


class Mnet(Base):
    """
    From whitenblackbox paper. This convnet structure cover many LeNet variants.
    """

    def __init__(self, sample_dimensions, n_classes, model_config):
        super().__init__(sample_dimensions, n_classes, model_config)

        self._act = return_act(self.config["net"]["act"])
        self._ks = self.config["net"]["ks"]

        self._conv1 = nn.Conv2d(sample_dimensions[0], 10, kernel_size=self._ks)
        self._conv2 = nn.Conv2d(10, 20, kernel_size=self._ks)

        conv_iter = []
        for _ in range(self.config["net"]["n_conv"] - 2):
            conv_iter.append(ConvBlock(
                20, 20, self._ks, padding=int((self._ks - 1) / 2)))
        self._conv_iter = nn.Sequential(*conv_iter)

        if "max" in self.config["net"]["pool"]:
            stride = int(self.config["net"]["pool"].split('_')[1])
            self._pool = lambda a: F.max_pool2d(a, stride)
            pool_factor = 2
        else:
            self._pool = lambda a: a
            pool_factor = 1

        self._fc_feat_dim = int((
                                        (
                                                (
                                                        (
                                                                (
                                                                        sample_dimensions[1]
                                                                        -
                                                                        self._ks + 1) // pool_factor
                                                        ) - self._ks + 1) //
                                                pool_factor
                                        ) ** 2
                                ) * 20)

        self._fc1 = nn.Linear(self._fc_feat_dim, 50)

        if self.config["net"]["drop"] != "none":
            drop = return_drop(self.config["net"]["drop"])
            self._drop = lambda a: drop(a)
        else:
            self._drop = lambda a: a

        fc_iter = []
        for _ in range(self.config["net"]["n_fc"] - 2):
            fc_iter.append(
                LinearBlock(50, 50, self.config["net"]["drop"]))
        self._fc_iter = nn.Sequential(*fc_iter)
        self._fc_final = nn.Linear(50, n_classes)

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
