import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.activethief import ActiveThief
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset

NUM_CLASSES = 2


class Ember2018(nn.Module):
    def __init__(self, model_dir, seed):
        super().__init__()
        model_file = Path(model_dir).joinpath("ember_model_2018.txt").__str__()
        self.ember = lgb.Booster(params={"seed": seed}, model_file=model_file)

    def forward(self, x):
        x = x.detach().cpu().numpy()
        y_preds = self.ember.predict(x)

        y_preds = torch.from_numpy(y_preds)

        return y_preds


class EmberSubsitute(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(in_features=2381, out_features=2400),
                       nn.ReLU(), nn.Dropout()])
        layers.extend([nn.Linear(in_features=2400, out_features=1200),
                       nn.ReLU(), nn.Dropout()])
        layers.extend([nn.Linear(in_features=1200, out_features=1200),
                       nn.ReLU()])
        layers.extend([nn.Linear(in_features=1200, out_features=1)])

        self.model = nn.Sequential(*layers)
        self._scaler = scaler

    def forward(self, x):
        # Add GPU support
        x = torch.from_numpy(self._scaler.transform(x)).float()

        return self.model(x).squeeze()


def prepare_ember2018_data(data_dir):
    X_train_path = Path(data_dir).joinpath("X_train.dat")
    y_train_path = Path(data_dir).joinpath("y_train.dat")
    y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
    N = y_train.shape[0]
    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r",
                        shape=(N, 2381))

    train_rows = (y_train == -1)  # read training dataset
    X_train = X_train[train_rows]
    y_train = y_train[train_rows]

    X_test_path = Path(data_dir).joinpath("X_test.dat")
    y_test_path = Path(data_dir).joinpath("y_test.dat")
    y_test = np.memmap(y_test_path, dtype=np.float32, mode="readwrite")
    N = y_test.shape[0]
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="readwrite",
                       shape=(N, 2381))

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    thief_dataset = CustomDataset(X_train, y_train)
    test_set = CustomDataset(X_test, y_test)

    return thief_dataset, test_set, scaler


if __name__ == '__main__':
    parser = ActiveThief.get_attack_args()
    parser.add_argument("--ember2018_data_dir", type=str,
                        help="Path to Ember2018 dataset")
    parser.add_argument("--ember2018_model_dir", type=str,
                        help="Path to Ember2018 dataset")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    thief_dataset, test_set, scaler = prepare_ember2018_data(
            args.ember2018_data_dir)

    victim_model = Ember2018(args.ember2018_model_dir, args.seed)
    substitute_model = EmberSubsitute(scaler)

    af = ActiveThief(victim_model, substitute_model, NUM_CLASSES,
                     args.iterations, args.selection_strategy,
                     args.victim_output_type, args.budget,
                     loss=torch.nn.BCEWithLogitsLoss())

    # Baset settings
    af.base_settings.save_loc = Path(args.save_loc)
    af.base_settings.gpus = args.gpus
    af.base_settings.num_workes = args.num_workers
    af.base_settings.batch_size = args.batch_size
    af.base_settings.seed = args.seed
    af.base_settings.deterministic = args.deterministic
    af.base_settings.debug = args.debug

    # Trainer settings
    af.trainer_settings.training_epochs = args.training_epochs
    af.trainer_settings.patience = args.patience
    af.trainer_settings.evaluation_frequency = args.evaluation_frequency
    af.trainer_settings.precision = args.precision
    af.trainer_settings.accuracy = args.accuracy

    af(thief_dataset, test_set)
