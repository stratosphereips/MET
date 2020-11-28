import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks import ActiveThief, AtlasThief
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import CustomDataset

NUM_CLASSES = 2


class Ember2018(nn.Module):
    def __init__(self, model_dir, seed):
        super().__init__()
        model_file = Path(model_dir).joinpath("ember_model_2018.txt").__str__()
        self.ember = lgb.Booster(params={"seed": seed}, model_file=model_file)

    def forward(self, x):
        y_preds = self.ember.predict(x.detach().cpu())

        y_preds = torch.from_numpy(y_preds)
        y_preds.to(x.device)

        return y_preds.unsqueeze(dim=-1)


class EmberSubsitute(nn.Module):
    def __init__(self, scaler, return_hidden):
        super().__init__()
        self._layer1 = nn.Sequential(
                nn.Linear(in_features=2381, out_features=2400), nn.ReLU(),
                nn.Dropout())
        self._layer2 = nn.Sequential(
                nn.Linear(in_features=2400, out_features=1200), nn.ReLU(),
                nn.Dropout())
        self._layer3 = nn.Sequential(
                nn.Linear(in_features=1200, out_features=1200), nn.ReLU())
        self._final = nn.Linear(in_features=1200, out_features=1)

        self._scaler = scaler
        self._return_hidden = return_hidden

    def forward(self, x):
        # TODO: create pytorch version of scaler
        x_scaled = self._scaler.transform(x.cpu().numpy())

        x_scaled = torch.from_numpy(x_scaled).float()
        x_scaled = x_scaled.to(x.device)

        hidden = self._layer3(self._layer2(self._layer1(x_scaled)))
        logits = self._final(hidden)

        return logits, hidden


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
    parser.add_argument("--atlasthief", action="store_true",
                        help="Use atlasthief for the attack (Default: False)")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    thief_dataset, test_set, scaler = prepare_ember2018_data(
            args.ember2018_data_dir)

    victim_model = Ember2018(args.ember2018_model_dir, args.seed)
    substitute_model = EmberSubsitute(scaler, args.atlasthief)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    if args.atlasthief:
        af = AtlasThief(victim_model, substitute_model, NUM_CLASSES,
                        args.iterations, args.victim_output_type,
                        args.budget, loss=torch.nn.BCEWithLogitsLoss())
    else:
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
