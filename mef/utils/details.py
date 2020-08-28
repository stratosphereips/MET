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
class ModelDetails:
    net: NetworkDetails

    def __init__(self, net):
        if isinstance(net, dict):
            self.net = from_dict(NetworkDetails, net)
        else:
            self.net = net
