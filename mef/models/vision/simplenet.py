import torch
import torch.nn as nn
import torch.nn.functional as F

from mef.utils.pytorch.blocks import ConvBlock


class SimpleNet(nn.Module):
    """https://paperswithcode.com/paper/lets-keep-it-simple-using-simple"""

    def __init__(self, input_dimensions, num_classes, pool=False, drop=False):
        super().__init__()

        self._layers = self._make_layers(input_dimensions[0])

        self._pool = lambda a: a
        if pool:
            self._pool = lambda a: F.max_pool2d(a, kernel_size=a.size()[2:])

        self._drop = lambda a: a
        if drop:
            drop = nn.Dropout(p=0.1)
            self._drop = lambda a: drop(a)

        test_input = torch.zeros(1, input_dimensions[0], input_dimensions[1], input_dimensions[2])
        test_out = self._layers(test_input)
        n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self._fc_final = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self._layers(x)

        # Global Max Pooling
        x = self._pool(x)
        x = self._drop(x)

        x = x.view(x.size(0), -1)
        x = self._fc_final(x)
        return x

    def _make_layers(self, channels):

        # Conv1/64 (3x3/1/1)
        layers = [
            ConvBlock(in_channels=channels, out_channels=64, kernel_size=3, padding=1,
                      use_batch_norm=True)]

        # 3x Conv2/128 (3x3/1/1)
        for i in range(3):
            in_channels = 128 if i != 0 else 64
            layers.append(ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=3,
                                    padding=1, use_batch_norm=True))

        # Max-pooling 2/2
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d(p=0.1)])

        # 2x Conv5/128 (3x3/1/1) + Conv7/128 (3x3/1/1)
        for i in range(3):
            out_channels = 128 if i != 2 else 256
            layers.append(ConvBlock(in_channels=128, out_channels=out_channels, kernel_size=3,
                                    padding=1, use_batch_norm=True))

        # Max-pooling 2/2
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d(p=0.1)])

        # 2x Conv7/128 (3x3/1/1)
        for i in range(2):
            layers.append(ConvBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1,
                                    use_batch_norm=True))

        # Max-pooling 2/2
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.Dropout2d(p=0.1)])

        # Conv10 (3x3/1/1)
        layers.append(
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1,
                      use_batch_norm=True))

        # Max-pooling 2/2
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d(p=0.1)])

        # Conv11 (1x1/1/1)
        layers.append(
            ConvBlock(in_channels=512, out_channels=2048, kernel_size=1, use_batch_norm=True))

        # Conv12 (1x1/1/1)
        layers.append(
            ConvBlock(in_channels=2048, out_channels=256, kernel_size=1, use_batch_norm=True))

        # Max-pooling 2/2
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d(p=0.1)])

        # Conv13 (3x3/1/1)
        layers.append(
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1,
                      use_batch_norm=True))

        model = nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

        model.apply(init_weights)
        return model
