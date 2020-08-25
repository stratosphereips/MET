from dataclasses import dataclass

from dacite import from_dict


@dataclass
class NetworkDetails:
    name: str
    act: str
    drop: str
    pool: str
    ks: int
    n_conv: int
    n_fc: int


@dataclass
class OptimizerDetails:
    name: str
    batch_size: int
    epochs: int
    momentum: float
    lr: float


@dataclass
class LossDetails:
    name: str


@dataclass
class ModelDetails:
    net: NetworkDetails
    opt: OptimizerDetails
    loss: LossDetails

    def __init__(self, net, opt, loss):
        if isinstance(net, dict):
            self.net = from_dict(NetworkDetails, net)
        else:
            self.net = net
        if isinstance(opt, dict):
            self.opt = from_dict(OptimizerDetails, opt)
        else:
            self.opt = opt
        if isinstance(net, dict):
            self.loss = from_dict(LossDetails, loss)
        else:
            self.loss = loss
