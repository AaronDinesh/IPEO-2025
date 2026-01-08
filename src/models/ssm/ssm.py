from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ssm.encoders import MLPEncoder, RNNEncoder, ResNetEncoder


@dataclass
class SSMOutputs:
    logits: List[Tensor]
    states: List[Tensor]
    reg_loss: Tensor
    step_masks: List[Tensor]


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
        ts_channels: int,
        img_freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_species = num_species

        self.state_init = nn.Parameter(torch.zeros(1, state_dim))

        self.env_encoder = MLPEncoder(
            state_space_dim=state_dim,
            in_features=env_dim,
            hidden_layers=1,
            hidden_layer_dim=state_dim,
        )
        self.ts_encoder = RNNEncoder(
            state_space_dim=state_dim,
            in_features=ts_channels,
            hidden_layers=(state_dim,),
        )
        self.img_encoder = ResNetEncoder(
            state_space_dim=state_dim,
            freeze_resnet=img_freeze_backbone,
        )

        self.decoder = nn.Linear(state_dim, num_species)

    def forward(
        self,
        env: Optional[Tensor] = None,
        ts: Optional[Tensor] = None,
        img: Optional[Tensor] = None,
        env_mask: Optional[Tensor] = None,
        ts_mask: Optional[Tensor] = None,
        img_mask: Optional[Tensor] = None,
    ) -> SSMOutputs:
        batch = None
        for tensor in (env, ts, img):
            if tensor is not None:
                batch = tensor.shape[0]
                break
        if batch is None:
            raise ValueError("At least one modality tensor must be provided.")

        state = self.state_init.expand(batch, -1)
        logits: List[Tensor] = []
        states: List[Tensor] = []
        reg_terms: List[Tensor] = []
        masks: List[Tensor] = []

        if env is not None:
            m = env_mask.float() if env_mask is not None else torch.ones(batch, device=state.device)
            prev = state
            new_state = self.env_encoder(prev, env)
            reg = (new_state - prev).pow(2).mean()
            state = m.unsqueeze(1) * new_state + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        if ts is not None:
            m = ts_mask.float() if ts_mask is not None else torch.ones(batch, device=state.device)
            prev = state
            new_state = self.ts_encoder(prev, ts)
            reg = (new_state - prev).pow(2).mean()
            state = m.unsqueeze(1) * new_state + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        if img is not None:
            m = img_mask.float() if img_mask is not None else torch.ones(batch, device=state.device)
            prev = state
            new_state = self.img_encoder(prev, img)
            reg = (new_state - prev).pow(2).mean()
            state = m.unsqueeze(1) * new_state + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        reg_loss = torch.stack(reg_terms).mean() if reg_terms else torch.tensor(0.0, device=state.device)
        return SSMOutputs(logits=logits, states=states, reg_loss=reg_loss, step_masks=masks)
