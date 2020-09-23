from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Fbeta, RunningAverage
from torch.utils.data import DataLoader
from tqdm import tqdm

from mef.utils.pytorch.datasets import CustomLabelDataset
from ..attacks.base import Base
from ..utils.config import Configuration
from ..utils.pytorch.ignite.metrics import MacroAccuracy


@dataclass
class CopyCatConfig:
    method: List[str]
    training_epochs: int


class CopyCat(Base):

    def __init__(self, target_model, opd_model, copycat_model, test_dataset,
                 pd_dataset=None, npd_dataset=None, save_loc="./cache/copycat"):
        super().__init__()

        # Get CopyCat's configuration
        self._config = Configuration.get_configuration(CopyCatConfig, "attacks/copycat")
        self._device = "cuda" if self._test_config.gpu is not None else "cpu"
        self._save_loc = save_loc

        # Datasets
        self._td = test_dataset
        self._pdd = pd_dataset
        self._pdd_sl = None  # pdd with stolen labels
        self._npdd = npd_dataset
        self._npdd_sl = None  # npdd with stolen labels

        # Models
        self._target_model = target_model
        self._opd_model = opd_model
        self._copycat_model = copycat_model

        # Dataset information
        self._num_classes = self._target_model.num_classes

        if self._test_config.gpu is not None:
            self._target_model.cuda()
            self._opd_model.cuda()
            self._copycat_model.cuda()

        # Stolen labels
        self._sl_pd = None
        self._sl_npd = None

        # Check method
        if self._config.method.lower() not in ["npd", "pd", "npd+pd"]:
            self._logger.error("Copycats's method must be one of {npd, pd, npd+pd}")
            raise ValueError()

    def _get_stolen_labels(self, dataset):
        if self._device == "cuda":
            self._target_model.cuda()

        self._target_model.eval()
        loader = DataLoader(dataset, pin_memory=True, batch_size=self._test_config.batch_size,
                            num_workers=4)
        stolen_labes = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader, desc="Getting labels")):
                x, _ = batch
                if self._device == "cuda":
                    x = x.cuda()

                y_pred = self._target_model(x)
                stolen_labes.append(torch.argmax(y_pred.cpu(), dim=1))

        return torch.cat(stolen_labes)

    def _add_ignite_events(self, trainer, evaluator, eval_loader):
        # Evaluator events
        def score_function(engine):
            val_macro_acc = engine.state.metrics["macro_accuracy"]
            return val_macro_acc

        early_stop = EarlyStopping(patience=self._test_config.early_stop_tolerance,
                                   score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop)

        to_save = {'copycat_model': self._copycat_model}
        checkpoint_handler = Checkpoint(to_save, DiskSaver(self._save_loc, require_empty=False),
                                        filename_prefix="best", score_function=score_function,
                                        score_name="macro_accuracy",
                                        filename_pattern="{filename_prefix}_{name}_({score_name}="
                                                         "{score}).{ext}",
                                        global_step_transform=global_step_from_engine(trainer))
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        # Trainer events
        RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")
        ProgressBar().attach(trainer, ["avg_loss"])

        @trainer.on(Events.EPOCH_COMPLETED(every=self._test_config.evaluation_frequency))
        def log_validation_results(trainer):
            evaluator.run(eval_loader)
            metrics = evaluator.state.metrics
            trainer.logger.info("Test dataset results - Epoch: {}  Macro-averaged accuracy: {:.1f}%"
                                " F1-score: {:.3f}".format(trainer.state.epoch,
                                                           100 * metrics["macro_accuracy"],
                                                           metrics["f1beta-score"]))

        return checkpoint_handler

    def _training(self, model, training_data):
        # Prepare trainer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        loss_function = F.cross_entropy
        trainer = create_supervised_trainer(model, optimizer, loss_function, device=self._device)
        trainer.logger = self._logger

        # Prepare evaluator
        metrics = {
            "macro_accuracy": MacroAccuracy(self._num_classes),
            "f1-score": Fbeta(beta=1)
        }
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=self._device)
        evaluator.logger = self._logger

        eval_loader = DataLoader(dataset=self._td, batch_size=self._test_config.batch_size,
                                 pin_memory=True, num_workers=4)
        checkpoint_handler = self._add_ignite_events(trainer, evaluator, eval_loader)

        # Start trainer
        train_loader = DataLoader(dataset=training_data, shuffle=True,
                                  batch_size=self._test_config.batch_size, pin_memory=True,
                                  num_workers=4)
        trainer.run(train_loader, max_epochs=self._config.training_epochs)

        # Load best model and remove the file
        self._logger.info("Loading best model")
        to_load = {'copycat_model': self._copycat_model}
        checkpoint_fp = self._save_loc + '/' + checkpoint_handler.last_checkpoint
        checkpoint = torch.load(checkpoint_fp)
        Checkpoint.load_objects(to_load, checkpoint)
        Path.unlink(Path(checkpoint_fp))

        return checkpoint_fp

    def _get_final_metrics(self):
        val_metrics = {
            "macro_accuracy": MacroAccuracy(self._target_model.num_classes),
            "f1beta-score": Fbeta(beta=1)
        }
        test_loader = DataLoader(self._td, pin_memory=True, batch_size=self._test_config.batch_size,
                                 num_workers=4)
        models = dict(target_model=self._target_model, opd_model=self._opd_model,
                      copycat_model=self._copycat_model)
        results = dict()
        for name, model in models.items():
            evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self._device)
            evaluator.run(test_loader)

            macro_accuracy = evaluator.state.metrics["macro_accuracy"]
            f1_score = evaluator.state.metrics["f1beta-score"]

            self._logger.info("{} test data results: Macro-averaged accuracy: {:.1f}% F1-score: "
                              "{:.3f}".format(name.capitalize().replace('_', ' '),
                                              100 * macro_accuracy, f1_score))

            results[name] = (macro_accuracy, f1_score)

        perf_over_tn = 100 * (results["copycat_model"][0] / results["target_model"][0])
        self._logger.info("Performance over target network: {:.1f}%".format(perf_over_tn))
        perf_over_opd = 100 * (results["copycat_model"][0] / results["opd_model"][0])
        self._logger.info("Performance over PD-OL network: {:.1f}%".format(perf_over_opd))

        return

    def run(self):
        self._logger.info("########### Starting CopyCat attack ###########")

        final_model = None
        if "npd" in self._config.method.lower():
            self._logger.info("Getting stolen labels for NPD dataset")
            self._logger.info("NPD dataset size: {}".format(len(self._npdd)))
            self._sl_npd = self._get_stolen_labels(self._npdd)
            self._npdd_sl = CustomLabelDataset(self._npdd, self._sl_npd)

            self._logger.info("Training copycat model with NPD-SL")
            final_model = self._training(self._copycat_model, self._npdd_sl)

        if "pd" in self._config.method.lower():
            self._logger.info("Getting stolen labels for PD dataset")
            self._logger.info("PD dataset size: {}".format(len(self._pdd)))
            self._sl_pd = self._get_stolen_labels(self._pdd)
            self._pdd_sl = CustomLabelDataset(self._pdd, self._sl_pd)

            self._logger.info("Training copycat model with PD-SL")
            final_model = self._training(self._copycat_model, self._pdd_sl)

        self._logger.info("Getting final metrics")
        self._get_final_metrics()

        self._logger.info("Saving final model to: {}".format(final_model))
        torch.save(dict(state_dict=self._copycat_model.state_dict), final_model)

        return
