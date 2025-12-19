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
    def forward(self, state: Tensor) -> Tensor:
        pass


class ClassificationDecoder(AbstractSSMDecoder):
    def __init__(
        self,
        state_space_dim: int,
        in_features: int,
        hidden_layers: int,
        hidden_layer_dim: int,
        activation_func: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__(state_space_dim)
        # Start with the input layer
        self.layers: list[nn.Module] = [
            nn.Linear(in_features=in_features, out_features=hidden_layer_dim),
            activation_func,
        ]

        # Add the hidden layers
        for _ in range(hidden_layer_dim):
            self.layers.append(
                nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim)
            )

            self.layers.append(activation_func)

        # Add the final output layer
        self.output_layer = nn.Linear(
            in_features=hidden_layer_dim + state_space_dim, out_features=state_space_dim
        )

        # Build the encoder
        self.encoder = nn.Sequential(*self.layers)

    def forward(self, state: Tensor) -> Tensor:
        x = self.encoder(state)
        return self.output_layer(x)
