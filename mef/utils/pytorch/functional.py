import torch
import torch.nn.functional as F


# Cross entropy for soft-labels
def soft_cross_entropy(logits, targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- targets * F.log_softmax(logits, dim=-1) *
                                    weights, dim=-1))
    else:
        return torch.mean(torch.sum(- targets * F.log_softmax(logits, dim=-1),
                                    dim=-1))


def get_prob_dist(logits):
    if len(logits.size()) == 1:
        return F.sigmoid(logits)
    else:
        return F.softmax(logits, dim=-1)


def get_labels(logits):
    if len(logits.size()) == 1:
        return torch.round(logits)
    else:
        return torch.argmax(logits, dim=-1)
