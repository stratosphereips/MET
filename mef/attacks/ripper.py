import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mef.attacks.base import AttackSettings, Base
from mef.utils.pytorch.datasets import GeneratorRandomDataset


@dataclass
class RipperSettings(AttackSettings):
    latent_dim: int
    greyscale: bool
    generated_data: str
    output_type: str
    budget: int

    def __init__(self,
                 latent_dim: int,
                 greyscale: bool,
                 generated_data: str,
                 output_type: str,
                 budget: int):
        self.latent_dim = latent_dim
        self.greyscale = greyscale
        self.generated_data = generated_data
        self.output_type = output_type.lower()
        self.budget = budget

        # Check configuration
        if self.generated_data not in ["random", "optimized"]:
            raise ValueError("Ripper's generated_data must be one of {random, "
                             "optimized}")


class Ripper(Base):
    def __init__(self,
                 victim_model,
                 substitute_model,
                 generator,
                 latent_dim,
                 greyscale=False,
                 generated_data="random",
                 output_type="softmax",
                 budget=20000):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        loss = F.cross_entropy

        attack_settings = RipperSettings(latent_dim, greyscale, generated_data,
                                         output_type, budget)
        super().__init__(victim_model, substitute_model, optimizer, loss)
        self.attack_settings = attack_settings

        # Ripper's specific attributes
        self._generator = generator

    @classmethod
    def get_attack_args(cls):
        parser = argparse.ArgumentParser(description="Ripper attack")
        parser.add_argument("--generated_data", default="random", type=str,
                            help="Type of generated data from generator. Can "
                                 "be one of {random, optimized} (Default: "
                                 "random)")
        parser.add_argument("--output_type", default="label", type=str,
                            help="Type of output from victim model {softmax, "
                                 "logits, one_hot, labels} (Default: label)")
        parser.add_argument("--training_epochs", default=100, type=int,
                            help="Number of training epochs for substitute "
                                 "model (Default: 100)")

        cls._add_base_args(parser)

        return

    def _get_student_dataset(self):
        if self.attack_settings.generated_data == "random":
            return GeneratorRandomDataset(self._victim_model,
                                          self._generator,
                                          self.attack_settings.latent_dim,
                                          self.data_settings.batch_size,
                                          self.attack_settings.output_type,
                                          self.attack_settings.greyscale)
        else:
            raise NotImplementedError()

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)
        self._logger.info("########### Starting Ripper attack ##########")
        # Get budget of the attack
        self._logger.info("Ripper's attack budget: {}".format(
                    self.attack_settings.budget))

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        self._train_substitute_model(self._thief_dataset)

        self._finalize_attack()
