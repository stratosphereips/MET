import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from mef.utils.pytorch.blocks import return_optimizer, return_loss_function


class ModelTraining:
    _test_config = None
    _logger = None

    @classmethod
    def _train_step(cls, model, optimizer, loss_function, loader):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(loader):
            if cls._test_config.gpu is not None:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % cls._test_config.batch_log_interval == 0:
                cls._logger.info(
                    "Training {}/{} ({:.0f}%)\tLoss: {:.6f}".format(batch_idx * len(inputs),
                                                                    len(loader.sampler),
                                                                    100. * batch_idx / len(loader),
                                                                    loss.item()))

        return

    @classmethod
    def evaluate_model(cls, model, evaluation_data):
        cls._logger.info("Evaluating model")

        if cls._test_config.gpu is not None:
            model.cuda()

        model.eval()

        loader = DataLoader(dataset=evaluation_data, batch_size=128, num_workers=1, pin_memory=True)

        targets_list = []
        predictions_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                if cls._test_config.gpu is not None:
                    inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, dim=1)

                    predictions_list.append(predictions.cpu())
                    targets_list.append(targets.cpu())

                if batch_idx % cls._test_config.batch_log_interval == 0:
                    cls._logger.info("Evaluating {}/{} ({:.0f}%)".format(batch_idx * len(inputs),
                                                                         len(loader.sampler),
                                                                         100. * batch_idx / len(
                                                                             loader)))

        y_pred = torch.cat(predictions_list).numpy()
        y_true = torch.cat(targets_list).numpy()
        metrics = classification_report(y_true, y_pred, output_dict=True)
        cls._logger.info('\n' + classification_report(y_true, y_pred))

        return metrics["accuracy"], metrics["macro avg"]["f1-score"]

    @classmethod
    def train_model(cls, model, training_data, validation_data=None,
                    evaluation_data=None):
        cls._logger.debug("Starting model training")

        if cls._test_config.gpu is not None:
            model.cuda()

        epochs = model.details.opt.epochs
        optimizer = return_optimizer(model, model.details.opt)
        loss_function = return_loss_function(model.details.loss)
        training_loader = DataLoader(dataset=training_data, shuffle=True,
                                     batch_size=model.details.opt.batch_size, num_workers=1,
                                     pin_memory=True)

        best_accuracy = None
        best_f1score = None
        for epoch in range(epochs):
            cls._logger.info("Epoch: {}".format(epoch))

            cls._train_step(model, optimizer, loss_function, training_loader)

            if evaluation_data is not None:
                if epoch % cls._test_config.evaluation_frequency:
                    best_accuracy, best_f1score = cls.evaluate_model(model, evaluation_data)

        return best_accuracy, best_f1score
