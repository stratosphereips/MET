from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttackSettings:
    pass


@dataclass
class TrainerSettings:
    training_epochs: int
    patience: int
    evaluation_frequency: int
    precision: int
    use_accuracy: bool


@dataclass
class BaseSettings:
    save_loc: Path
    gpu: int
    num_workers: int
    batch_size: int
    seed: int
    deterministic: bool
    debug: bool
