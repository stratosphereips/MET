import torch
import torch.nn.functional as F


# Cross entropy for soft-labels
def soft_cross_entropy(y_hat, y_output, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- y_output * F.log_softmax(y_hat, dim=1) *
                                    weights, dim=1))
    else:
        return torch.mean(torch.sum(- y_output * F.log_softmax(y_hat, dim=1),
                                    dim=1))
