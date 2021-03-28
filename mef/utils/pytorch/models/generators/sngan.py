from typing import Callable, Optional

import torch.nn as nn

from torch_mimicry.nets import sngan

sngan_types = {
    32: sngan.SNGANGenerator32,
    48: sngan.SNGANGenerator48,
    64: sngan.SNGANGenerator64,
    128: sngan.SNGANGenerator128,
}


class Sngan(nn.Module):
    def __init__(
        self,
        checkpoint_file: str,
        resolution: int,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        if resolution not in [32, 48, 64, 128]:
            raise ValueError("SNGAN resolution must be one of [32, 48, 64, 128].")

        self._generator = sngan_types[resolution]()
        self._latent_dim = 128
        self._generator.restore_checkpoint(checkpoint_file)
        self._transform = transform

    def cuda(self):
        self.to("cuda")
        self._generator.cuda()

    def forward(self, z):
        x = self._generator(z)

        if self._transform is not None:
            x = self._transform(x)

        return x
