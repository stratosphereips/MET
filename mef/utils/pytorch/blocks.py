import torch.nn as nn


def return_act(activation, inplace=False):
    if activation == "relu":
        act_func = nn.ReLU(inplace=inplace)
    elif activation == "elu":
        act_func = nn.ELU(inplace=inplace)
    elif activation == "prelu":
        act_func = nn.PReLU()
    elif activation == "tanh":
        act_func = nn.Tanh()
    else:
        raise ValueError(
                "Activation type should be one of {relu, elu, prelu, tanh}.")

    return act_func


def return_batch_norm(type, input_channels, eps, momentum, affine=True):
    if type == "2d":
        batch_norm = nn.BatchNorm2d(input_channels, eps, momentum, affine)
    else:
        raise ValueError("Batch normalization type must be one of {2d}.")

    return batch_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 use_batch_norm=False, eps=1e-05, momentum=0.05, affine=True,
                 activation="relu", padding_mode="zeros"):
        super().__init__()
        self._use_batch_norm = use_batch_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias,
                              padding_mode)
        self.batch_norm = None
        if self._use_batch_norm:
            self.batch_norm = return_batch_norm("2d", out_channels, eps=eps,
                                                momentum=momentum,
                                                affine=affine)
        self.act = return_act(activation, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self._use_batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        return x
