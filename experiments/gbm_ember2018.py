import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from met.attacks import ActiveThief
from met.utils.ios import mkdir_if_missing
from met.utils.pytorch.datasets import NumpyDataset
from met.utils.pytorch.lightning.module import TrainableModel, VictimModel

NUM_CLASSES = 2


class Ember2018(nn.Module):
    def __init__(self, model_dir, seed):
        super().__init__()
        model_file = Path(model_dir).joinpath("ember_model_2018.txt").__str__()
        self.ember = lgb.Booster(params={"seed": seed}, model_file=model_file)

    def forward(self, x):
        y_preds = self.ember.predict(x.detach().cpu().numpy())

        return torch.from_numpy(y_preds.astype(np.float32)).to(x.device)


class EmberSubsitute(nn.Module):
    def __init__(self, scaler, return_hidden=False):
        super().__init__()
        self._layer1 = nn.Sequential(
            nn.Linear(in_features=2381, out_features=2400),
            nn.ELU(),
            nn.LayerNorm(2400),
            nn.Dropout(p=0.2),
        )
        self._layer2 = nn.Sequential(
            nn.Linear(in_features=2400, out_features=1024),
            nn.ELU(),
            nn.LayerNorm(1024),
            nn.Dropout(p=0.2),
        )
        self._layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
        )
        self._layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
        )
        self._layer5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2),
        )
        self._final = nn.Linear(in_features=128, out_features=1)

        self._scaler = scaler
        self._return_hidden = return_hidden

    def forward(self, x):
        # TODO: create pytorch version of scaler
        x_scaled = self._scaler.transform(x.cpu().numpy()).astype(np.float32)

        x_scaled = torch.from_numpy(x_scaled)
        x_scaled = x_scaled.to(x.device)

        hidden = self._layer5(
            self._layer4(self._layer3(self._layer2(self._layer1(x_scaled))))
        )
        logits = self._final(hidden)

        if self._return_hidden:
            return logits, hidden
        else:
            return logits


def prepare_ember2018_data(data_dir):
    X_train_path = Path(data_dir).joinpath("X_train.dat")
    y_train_path = Path(data_dir).joinpath("y_train.dat")
    y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
    N = y_train.shape[0]
    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, 2381))

    train_rows = y_train == -1  # read training dataset
    X_train = X_train[train_rows]
    y_train = y_train[train_rows]

    X_test_path = Path(data_dir).joinpath("X_test.dat")
    y_test_path = Path(data_dir).joinpath("y_test.dat")
    y_test = np.memmap(y_test_path, dtype=np.float32, mode="readwrite").astype(np.int32)
    N = y_test.shape[0]
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="readwrite", shape=(N, 2381))

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    adversary_dataset = NumpyDataset(X_train, y_train)
    test_set = NumpyDataset(X_test, y_test)

    return adversary_dataset, test_set, scaler


if __name__ == "__main__":
    parser = ActiveThief.get_attack_args()
    parser.add_argument(
        "--ember2018_data_dir", type=str, help="Path to Ember2018 dataset"
    )
    parser.add_argument(
        "--ember2018_model_dir", type=str, help="Path to Ember2018 dataset"
    )
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    # Prepare data
    adversary_dataset, test_set, scaler = prepare_ember2018_data(args.ember2018_data_dir)

    # Prepare models
    victim_model = Ember2018(args.ember2018_model_dir, args.seed)
    victim_model = VictimModel(victim_model, NUM_CLASSES, "raw")
    substitute_model = EmberSubsitute(scaler)
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.Adam,
        torch.nn.BCEWithLogitsLoss(),
    )

    attack = ActiveThief(
        victim_model,
        substitute_model,
        args.selection_strategy,
        args.iterations,
        args.budget,
    )

    # Baset settings
    attack.base_settings.save_loc = Path(args.save_loc)
    attack.base_settings.gpu = args.gpu
    attack.base_settings.num_workers = args.num_workers
    attack.base_settings.batch_size = args.batch_size
    attack.base_settings.seed = args.seed
    attack.base_settings.deterministic = args.deterministic
    attack.base_settings.debug = args.debug

    # Trainer settings
    attack.trainer_settings.training_epochs = args.training_epochs
    attack.trainer_settings.precision = args.precision
    attack.trainer_settings.use_accuracy = args.accuracy

    attack(adversary_dataset, test_set)
