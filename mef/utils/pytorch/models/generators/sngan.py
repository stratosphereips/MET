import torch.nn as nn

from torch_mimicry.nets import sngan

sngan_types = {
    32: sngan.SNGANGenerator32,
    48: sngan.SNGANGenerator48,
    64: sngan.SNGANGenerator64,
    128: sngan.SNGANGenerator128,
}


class Sngan(nn.Module):
    def __init__(self, checkpoint_file: str, resolution: int, gpu: bool = False):
        super().__init__()

        if resolution not in [32, 48, 64, 128]:
            raise ValueError("SNGAN resolution must be one of [32, 48, 64, 128].")

        self._generator = sngan_types[resolution]()
        self._latent_dim = 128
        self._generator.restore_checkpoint(checkpoint_file)

        if gpu:
            self._generator.cuda()

    def forward(self, z):
        return self._generator(z)
