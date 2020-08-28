import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class MacroAccuracy(Metric):

    def __init__(self, num_classes, output_transform=lambda x: x):
        self._classes = range(num_classes)
        self._num_class_correct = None
        self._num_class_examples = None
        super().__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._num_class_correct = [0 for _ in self._classes]
        self._num_class_examples = [0 for _ in self._classes]
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)

        for class_ in self._classes:
            class_mask = (y == class_)
            y_class = y[class_mask]
            indices_class = indices[class_mask]
            correct_class = torch.eq(indices_class, y_class).view(-1)

            self._num_class_correct[class_] += torch.sum(correct_class).item()
            self._num_class_examples[class_] += correct_class.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if 0 in self._num_class_examples:
            raise NotComputableError('MacroAccuracy must have at least one example in each class '
                                     'before it can be computed.')
        total_acc = 0
        for class_ in self._classes:
            total_acc += self._num_class_correct[class_] / self._num_class_examples[class_]
        return total_acc / len(self._classes)
