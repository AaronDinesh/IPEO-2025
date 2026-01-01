from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import ResNet50_Weights, resnet50


@dataclass
class SSMOutputs:
    logits: List[Tensor]
    states: List[Tensor]
    reg_loss: Tensor
    step_masks: List[Tensor]


class ResidualMLP(nn.Module):
    """Lightweight residual MLP used for modality-specific encoders."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.proj = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(dropout))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.proj(x)


class ImageEncoder(nn.Module):
    """ResNet50 backbone with a small projection head."""

    def __init__(self, out_dim: int, freeze_backbone: bool = True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.features(x).flatten(1)
        return self.proj(feat)


class TimeSeriesEncoder(nn.Module):
    """Encode quarterly Landsat time series."""

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # dropout only applied when num_layers > 1
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        _, h_n = self.rnn(x)
        return self.proj(h_n[-1])


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

        self.env_encoder = ResidualMLP(env_dim, state_dim, state_dim, dropout=dropout)
        self.ts_encoder = TimeSeriesEncoder(ts_channels, state_dim, state_dim, dropout=dropout)
        self.img_encoder = ImageEncoder(state_dim, freeze_backbone=img_freeze_backbone)

        self.fuse_env = ResidualMLP(state_dim * 2, state_dim, state_dim, dropout=dropout)
        self.fuse_ts = ResidualMLP(state_dim * 2, state_dim, state_dim, dropout=dropout)
        self.fuse_img = ResidualMLP(state_dim * 2, state_dim, state_dim, dropout=dropout)

        self.decoder = nn.Linear(state_dim, num_species)

    def _fuse(self, state: Tensor, feat: Tensor, block: nn.Module) -> Tuple[Tensor, Tensor]:
        delta = block(torch.cat([state, feat], dim=1))
        new_state = state + delta
        reg = (delta.pow(2).mean())
        return new_state, reg

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
            env_feat = self.env_encoder(env)
            fused, reg = self._fuse(state, env_feat, self.fuse_env)
            state = m.unsqueeze(1) * fused + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        if ts is not None:
            m = ts_mask.float() if ts_mask is not None else torch.ones(batch, device=state.device)
            prev = state
            ts_feat = self.ts_encoder(ts)
            fused, reg = self._fuse(state, ts_feat, self.fuse_ts)
            state = m.unsqueeze(1) * fused + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        if img is not None:
            m = img_mask.float() if img_mask is not None else torch.ones(batch, device=state.device)
            prev = state
            img_feat = self.img_encoder(img)
            fused, reg = self._fuse(state, img_feat, self.fuse_img)
            state = m.unsqueeze(1) * fused + (1 - m.unsqueeze(1)) * prev
            logits.append(self.decoder(state))
            reg_terms.append(reg * m.mean())
            states.append(state)
            masks.append(m)

        reg_loss = torch.stack(reg_terms).mean() if reg_terms else torch.tensor(0.0, device=state.device)
        return SSMOutputs(logits=logits, states=states, reg_loss=reg_loss, step_masks=masks)
