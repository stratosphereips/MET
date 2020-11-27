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
    if logits.size()[-1] == 1:
        sig_output = torch.sigmoid(logits)
        return torch.stack(list(
            map(lambda y: torch.tensor([1 - y, y]), sig_output)))
    else:
        return torch.softmax(logits, dim=-1)


def get_labels(output):
    if output.size()[-1] == 1:
        return torch.round(output)
    else:
        return torch.argmax(output, dim=-1)
