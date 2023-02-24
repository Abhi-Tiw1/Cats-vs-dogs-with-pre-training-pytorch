import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class Regularize:
    def __init__(self, **kwargs):
        self.lambda_ = kwargs['lambda']
        self.state = kwargs['state']
        self.norm_type = kwargs['norm_type']

    def reg(self, model, loss):
        if self.state:
            if self.norm_type == 'l2':
                l_norm = sum(p.pow(2).sum() for p in model.parameters())
            elif self.norm_type == 'l1':
                l_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.lambda_ * l_norm
        else:
            return loss

        return loss
