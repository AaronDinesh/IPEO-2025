from abc import ABC, abstractmethod
from typing import Callable, Optional, Required, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import ResNet50_Weights, resnet50

from src.models.ssm.decoders import AbstractSSMDecoder


class AbstractSSMEncoder(nn.Module, ABC):
    """
    An Abstract Encoder class for the SSM model
    """

    def __init__(self, state_space_size: int):
        super(AbstractSSMEncoder, self).__init__()
        self.state_space_dim = state_space_size

    @abstractmethod
    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        pass


class ResNetEncoder(AbstractSSMEncoder):
    def __init__(
        self,
        state_space_dim: int,
        activation_func: Callable = F.relu,
        freeze_resnet: bool = False,
    ):
        super().__init__(state_space_dim)
        self.resnetModel = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_feature_dim = self.resnetModel.fc.in_features
        self.activation_func = activation_func
        # Removing the Final Linear Classification Layer
        self.resnetModel.fc = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]

        if freeze_resnet:
            for param in self.resnetModel.parameters():
                param.requires_grad = False

        # Final layer to encode into the state tensor
        self.linear_layer = nn.Linear(
            in_features=self.image_feature_dim + self.state_space_dim,
            out_features=self.state_space_dim,
        )

    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        # Compute the image features
        x = self.resnetModel(input)
        # Maybe we can remove this call to activation_func
        x = self.activation_func(x)
        # Concatenate the state (maybe there is a better way to do this?)
        x = torch.cat([x, state], dim=1)
        # Output the new state after encoding the current modality
        return self.linear_layer(x)


class MLPEncoder(AbstractSSMEncoder):
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

    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        x = self.encoder(input)
        x = torch.cat([x, state], dim=1)
        return self.output_layer(x)


class RNNEncoder(AbstractSSMEncoder):
    def __init__(
        self,
        state_space_dim: int,
        in_features: int,
        hidden_layers: Tuple[int],
        activation: Callable = F.relu,
    ):
        super().__init__(state_space_dim)

        self.activation = activation

        dim_layers = (
            [in_features]
            + list(hidden_layers)
            + [
                self.state_space_dim,
            ]
        )

        self.layers = nn.ModuleList()
        for i, (inDim, outDim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers) - 2:
                self.layers.append(nn.RNN(inDim + self.state_space_dim, outDim, batch_first=True))
            else:
                self.layers.append(nn.RNN(inDim, outDim, batch_first=True))

    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            out, h_n = layer(input)
            input = self.activation(out)

        output, h_n = self.layers[-1](torch.cat([input, state], dim=1))

        return output

class LSTMEncoder(AbstractSSMEncoder):
    def __init__(
        self,
        state_space_dim: int,
        in_features: int,
        hidden_layers: Tuple[int],
        activation: Callable = F.relu,
    ):
        super().__init__(state_space_dim)

        self.activation = activation

        dim_layers = [in_features] + list(hidden_layers) + [self.state_space_dim,]

        self.layers = nn.ModuleList()
        for i, (inDim, outDim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers)-2:
                self.layers.append(nn.LSTM(inDim + self.state_space_dim, outDim, batch_first=True))
            else:
                self.layers.append(nn.LSTM(inDim, outDim, batch_first=True))

    def forward(self, state: Tensor, input: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            out, tups = layer(input)
            input = self.activation(out)

        output, tups = self.layers[-1](torch.cat([input, state], dim=1))

        return output
