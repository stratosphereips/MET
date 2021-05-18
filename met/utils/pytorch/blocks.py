from dataclasses import dataclass


@dataclass
class ConvBlock:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0


@dataclass
class MaxPoolLayer:
    kernel_size: int
    stride: int
    padding: int = 0
