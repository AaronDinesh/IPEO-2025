from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ssm.encoders import MLPEncoder, ResNetEncoder, RNNEncoder


@dataclass
class SSMOutputs:
    logits: List[Tensor]
    states: List[Tensor]
    reg_loss: Tensor


class StateSpaceModel(nn.Module):
    """
    Multi-modal state space model (MultiModN-style) that fuses tabular climate,
    Landsat time series, and Sentinel image patches into a shared state.
    """

    def __init__(
        self,
        num_species: int,
        state_dim: int,
        env_dim: int,
        time_series_channels: int,
        img_freeze_backbone: bool = True,
        img_backbone: str = "resnet50",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_species = num_species

        self.state_init = nn.Parameter(torch.randn(1, state_dim))

        self.env_encoder = MLPEncoder(
            state_space_dim=state_dim,
            in_features=env_dim,
            hidden_layers=2,
            hidden_layer_dim=state_dim,
        )
        self.ts_encoder = RNNEncoder(
            state_space_dim=state_dim,
            in_features=time_series_channels,
            hidden_layers=(state_dim,),
        )
        self.img_encoder = ResNetEncoder(
            state_space_dim=state_dim,
            freeze_resnet=img_freeze_backbone,
            backbone=img_backbone,
        )

        self.decoder = nn.Linear(state_dim, num_species)

    def forward(
        self,
        env: Tensor,
        ts: Tensor,
        img: Tensor,
    ) -> SSMOutputs:
        batch = env.shape[0]
        state = self.state_init.expand(batch, -1)
        logits: List[Tensor] = []
        states: List[Tensor] = []
        reg_terms: List[Tensor] = []

        modalities = (
            (self.env_encoder, env),
            (self.ts_encoder, ts),
            (self.img_encoder, img),
        )

        for encoder, data in modalities:
            prev_state = state
            new_state = encoder(prev_state, data)
            # Computing the state change for the penalty
            reg = (new_state - prev_state).pow(2).mean()
            state = new_state
            logits.append(self.decoder(state))
            reg_terms.append(reg)
            states.append(state)

        reg_loss = torch.stack(reg_terms).mean()
        return SSMOutputs(logits=logits, states=states, reg_loss=reg_loss)
