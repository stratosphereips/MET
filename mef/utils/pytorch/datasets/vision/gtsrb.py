import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GTSRB(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        root = Path(root)
        self._check_dataset_existence(root)

        if train:
            root = root.joinpath("train_images")
            csv_file = "train.csv"
            set_type = "train"
        else:
            root = root.joinpath("test_images")
            csv_file = "test.csv"
            set_type = "test"

        csv_data_info = pd.read_csv(root.joinpath(csv_file))
        self.samples = []
        self.targets = []
        for idx in range(len(csv_data_info)):
            self.samples.append(str(root.joinpath(csv_data_info.iloc[idx, 0])))
            self.targets.append(csv_data_info.iloc[idx, 1])

        self._transform = transform
        self._target_transform = target_transform

        print(
            f"Loaded {self.__class__.__name__} ({set_type}) with {len(self.samples)} samples"
        )

    def _check_dataset_existence(self, root: Path):
        train_folder = root.joinpath("train_images")
        test_folder = root.joinpath("test_images")

        if not train_folder.is_dir() or not test_folder.is_dir():
            raise (
                RuntimeError(
                    f"GTSRB dataset not found at {root}, please download them from https://benchmark.ini.rub.de/gtsrb_news.html"
                )
            )

        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = Image.open(self.samples[idx])
        target = self.targets[idx]

        if self._transform is not None:
            sample = self._transform(sample)

        if self._target_transform is not None:
            target = self._target_transform()

        return sample, target
