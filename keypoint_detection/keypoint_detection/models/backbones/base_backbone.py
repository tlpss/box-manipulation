import abc

from torch import nn as nn


class Backbone(nn.Module, abc.ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abc.abstractmethod
    def get_n_channels_out(self):
        pass
