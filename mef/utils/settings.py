from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttackSettings:
    pass


@dataclass
class TrainerSettings:
    training_epochs: int = 1000
    patience: int = None
    evaluation_frequency: int = None
    precision: int = 32
    use_accuracy: bool = False


@dataclass
class BaseSettings:
    save_loc: Path = Path("./cache/")
    gpu: bool = False
    num_workers: int = 1
    batch_size: int = 32
    seed: int = None
    deterministic: bool = False
    debug: bool = False
