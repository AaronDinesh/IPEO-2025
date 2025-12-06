from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class AbstractSSMDecoder(nn.Module, ABC):
    """
    An Abstract Encoder class for the SSM model
    """

    def __init__(self, state_space_size: int):
        super(AbstractSSMDecoder, self).__init__()
        self.state_space_size = state_space_size

    @abstractmethod
    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        pass
