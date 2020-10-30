import torch.nn as nn
from torch import optim


def return_optimizer(model, optimizer_config):
    """Prepares optimizer for the model.

    Args:
        model (Mnet/Nin): Instance of the training model
        optimizer_config (OptimizerDetails): Configuration for the optimizer

    Raises:
        ValueError: Raised if the optimizer type is not one of
        {SGD, ADAM, RMSprop}

    Returns:
        SGD/Adam/RMSprop: Instance of the optimizer
    """

    optimizer_name = optimizer_config.name
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_config.lr,
                              momentum=optimizer_config.momentum)
    elif optimizer_name == "ADAM":
        optimizer = optim.Adam(model.parameters(),
                               lr=optimizer_config.lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=optimizer_config.lr,
                                  momentum=optimizer_config.momentum)
    else:
        raise ValueError(
                "Optimizer name should be one of {SGD, ADAM, RMSprop}.")

    return optimizer


def return_loss_function(loss_function_name):
    if loss_function_name == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    elif loss_function_name == "mse":
        loss_function = nn.MSELoss()
    else:
        raise ValueError(
                "Loss function name for should be on of {cross_entropy, mse}")

    return loss_function


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


def return_drop(type, p=0.5, inplace=False):
    """Returns instance of dropout type.

    Args:
        drop (str): Name of the dropout

    Raises:
        ValueError: Raised if the dropout name is not 'normal'

    Returns:
        dropout(): Instance of the dropout function.
    """
    if type == "normal":
        drop_func = nn.Dropout(p=p, inplace=inplace)
    elif type == "2d":
        drop_func = nn.Dropout2d(p=p, inplace=inplace)
    else:
        raise ValueError("Dropout type must be one of {normal, 2d}.")
    return drop_func


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


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 act="relu", dropout="none"):
        super().__init__()
        self._conv = nn.Linear(in_features, out_features, bias=bias)
        self._act = return_act(act, inplace=True)
        self._dropout = nn.Dropout(inplace=True)
        self._dropout_option = dropout

    def forward(self, x):
        x = self._conv(x)
        x = self._act(x)
        if self._dropout_option == 'normal':
            x = self._dropout(x)
        return x


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer=2, hidden_dim=10,
                 activation="relu"):
        super().__init__()
        self._n_layer = n_layer
        self._hidden_dim = hidden_dim
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._act = return_act(activation, inplace=True)

        self._fc1 = nn.Linear(self._input_dim, self._hidden_dim)
        fc_iter = []
        for _ in range(self._n_layer - 2):
            fc_iter.append(LinearBlock(self._hidden_dim, self._hidden_dim))
        self._fc_iter = nn.Sequential(*fc_iter)
        self._fc_final = nn.Linear(self._hidden_dim, self._output_dim)

    def forward(self, x):
        x = self._act(self._fc1(x))
        x = self._fc_iter(x)
        x = self._fc_final(x)
        return x
