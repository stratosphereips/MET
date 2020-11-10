import torch
import torch.nn.functional as F

from mef.attacks.base import Base
from mef.utils.pytorch.datasets import GeneratorRandomDataset


class Ripper(Base):
    def __init__(self, victim_model, substitute_model, generator,
                 latent_dim, greyscale=False, generated_data="random",
                 budget=20000, output_type="softmax", training_epochs=1000,
                 batch_size=64, save_loc="./cache/ripper", gpus=0, seed=None,
                 deterministic=True, debug=False, precision=32,
                 accuracy=False):
        optimizer = torch.optim.Adam(substitute_model.parameters())
        loss = F.cross_entropy

        super().__init__(victim_model, substitute_model, optimizer, loss,
                         training_epochs=training_epochs,
                         batch_size=batch_size, save_loc=save_loc,
                         validation=False, gpus=gpus, seed=seed,
                         deterministic=deterministic, debug=debug,
                         precision=precision, accuracy=accuracy)

        # Ripper's specific attributes
        self._generated_data = generated_data
        self._output_type = output_type
        self._budget = budget
        self._generator = generator
        self._latent_dim = latent_dim
        self._greyscale = greyscale

        # Check configuration
        if self._generated_data not in ["random", "optimized"]:
            self._logger.error(
                    "Ripper's generated_data must be one of {random, "
                    "optimized}")
            raise ValueError()

    def _get_student_dataset(self):
        if self._generated_data == "random":
            return GeneratorRandomDataset(self._victim_model, self._generator,
                                          self._latent_dim, self._batch_size,
                                          self._output_type, self._greyscale)
        else:
            raise NotImplementedError()

    def run(self, *args, **kwargs):
        self._parse_args(args, kwargs)
        self._logger.info(
                "########### Starting Ripper attack ###########")
        # Get budget of the attack
        self._logger.info(
                "Ripper's attack budget: {}".format(self._budget))

        # For consistency between attacks the student dataset is called
        # thief dataset
        self._thief_dataset = self._get_student_dataset()

        self._train_model(self._substitute_model, self._optimizer,
                          self._thief_dataset)

        self._finalize_attack()
