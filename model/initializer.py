import torch.nn as nn


def init_weights(m, initial):
    if type(m) == nn.Linear:
        initial(m.weight)
        m.bias.data.fill_(0.01)
    return
