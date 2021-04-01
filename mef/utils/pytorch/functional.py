from typing import Optional

import torch
import torch.nn.functional as F


# Cross entropy for soft-labels
def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weights is not None:
        return torch.mean(
            torch.sum(-targets * F.log_softmax(logits, dim=-1) * weights, dim=-1)
        )
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1))


def get_prob_vector(logits: torch.Tensor) -> torch.Tensor:
    if logits.size()[-1] == 1:
        sig_output = torch.sigmoid(logits)
        return torch.stack(list(map(lambda y: torch.tensor([1 - y, y]), sig_output)))
    else:
        return torch.softmax(logits, dim=-1)


def get_class_labels(input: torch.Tensor) -> torch.Tensor:
    # If dtype is not float the input already consists of class labels
    if not input.is_floating_point():
        return input

    if input.ndim == 1:
        return torch.round(input)
    else:
        return torch.argmax(input, dim=-1)
