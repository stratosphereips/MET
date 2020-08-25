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
        self.net = from_dict(NetworkDetails, net)
        self.opt = from_dict(OptimizerDetails, opt)
        self.loss = from_dict(LossDetails, loss)
