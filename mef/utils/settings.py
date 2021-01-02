from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttackSettings:
    pass


@dataclass
class TrainerSettings:
    training_epochs: int = 1000
    patience: int = 100
    evaluation_frequency: int = 1
    precision: int = 32
    accuracy: bool = False


@dataclass
class BaseSettings:
    save_loc: Path = Path("./cache/")
    gpus: int = 0
    num_workers: int = 1
    batch_size: int = 32
    seed: int = None
    deterministic: bool = True
    debug: bool = False
