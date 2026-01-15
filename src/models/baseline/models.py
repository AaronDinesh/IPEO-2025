from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ClimateVarsEncoder, TimeSeriesEncoder, ResNetEncoder
from .decoder import SpeciesDecoder
from torch.utils.data import Dataset
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------
# Dataset

class MultiModalDataset(Dataset):
    """
    Custom dataset for multi-modal dataset
    """
    def __init__(self, env_vars: torch.Tensor, ts_data: torch.Tensor, images: torch.Tensor, labels: torch.Tensor):
        self.env_vars = env_vars.astype(np.float32)
        self.ts_data = ts_data.astype(np.float32)
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.float32)
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        return self.env_vars[idx], self.ts_data[idx], self.images[idx], self.labels[idx]
    
# ----------------------------------------------------------------------------------------------------------------------------------
# Models

class MultiModalFuserBase(nn.Module):
    """
    Parent class for fusion models
    """
    def __init__(self, env_input_dim, landsat_input_dim, hidden_dim, embed_dim, dropout: float=0, rnn_layers: int=1, resnet_model: int=18):
        super().__init__()
        # Instantiate encoders for each modality
        self.env_encoder = ClimateVarsEncoder(env_input_dim, hidden_dim, embed_dim, dropout=dropout)
        self.time_encoder = TimeSeriesEncoder(landsat_input_dim, embed_dim, rnn_layers, dropout)
        self.image_encoder = ResNetEncoder(embed_dim, type=resnet_model)
        
        # Classification head
        self.decoder = SpeciesDecoder(embed_dim, hidden_dim)

class MultiModalSimpleFuser(MultiModalFuserBase):
    """
    Baseline model for fusing modalities
    """
    def __init__(self, env_input_dim, landsat_input_dim, hidden_dim, embed_dim, dropout=0, rnn_layers=1, use_sum=True, resnet_model: int=18):
        super().__init__(env_input_dim, landsat_input_dim, hidden_dim, embed_dim, dropout, rnn_layers, resnet_model)
                
        # Fusion type
        self.use_sum = use_sum
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        # Encode modalities
        x = self.env_encoder(x)
        y = self.time_encoder(y)
        z = self.image_encoder(z)
        
        # Fuse modalities
        if self.use_sum:
            output = x + y + z
        else:
            output = torch.cat((x,y,z), dim=-1)
            
        # Binary classification for each plant species
        return self.decoder(output)
    
class MultiModalAttentionFuser(MultiModalFuserBase):
    """
    Model using self attention as a fusing mechanism
    """
    def __init__(self, env_input_dim, landsat_input_dim, hidden_dim, embed_dim, dropout=0, rnn_layers=1, resnet_model: int=18):
        super().__init__(env_input_dim, landsat_input_dim, hidden_dim, embed_dim, dropout, rnn_layers, resnet_model)
        
        self.attn_pooling = LearnedAggregation(embed_dim, dropout=dropout)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        # Encode modalities
        x = self.env_encoder(x)
        y = self.time_encoder(y)
        z = self.image_encoder(z)
        
        # Fuse modalities
        tokens = torch.stack((x,y,z), dim=1)
        output = self.attn_pooling(tokens)
        
        # Binary classification for each plant species
        return self.decoder(output)

# ----------------------------------------------------------------------------------------------------------
# Attention pooling code provided by https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling#learned-aggregation

class AttentionPool2d(nn.Module):
    """
    Attention for Learned Aggregation
    """
    def __init__(self,
        embed_dim: int,
        bias: bool=True,
        norm: Callable[[int], nn.Module]=nn.LayerNorm,
        dropout: float=0.0
    ):
        super().__init__()
        self.norm = norm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.vk = nn.Linear(embed_dim, embed_dim*2, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cls_q: torch.Tensor):
        x = self.norm(x)
        B, N, D = x.shape
        
        cls_q = cls_q.view(1, 1, -1)          # (1, 1, D)
        q = self.q(cls_q.expand(B, -1, -1))    # (B, 1, D)
        k, v = self.vk(x).reshape(B, N, 2, D).permute(2, 0, 1, 3).chunk(2, 0)

        attn = q @ k.transpose(-2, -1).squeeze(0)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        x = (attn @ v.squeeze(0)).transpose(1, 2).squeeze(-1) # shape (B,D)
        return self.proj(x)

class LearnedAggregation(nn.Module):
    """
    Learned Aggregation from https://arxiv.org/abs/2112.13692
    """
    def __init__(self,
        embed_dim: int,
        attn_bias: bool=True,
        ffn_expand: int|float=3,
        norm: Callable[[int], nn.Module]=nn.LayerNorm,
        act_cls: Callable[[None], nn.Module]=nn.GELU,
        dropout: float=0.0
    ):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(embed_dim))
        self.cls_q = nn.Parameter(torch.zeros(embed_dim))
        self.attn = AttentionPool2d(embed_dim, attn_bias, norm)
        self.norm = norm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*ffn_expand)),
            act_cls(),
            nn.Dropout(p=dropout),
            nn.Linear(int(embed_dim*ffn_expand), embed_dim),
            nn.Dropout(p=dropout),
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.cls_q + self.gamma_1 * self.attn(x, self.cls_q)
        return x + self.gamma_2 * self.ffn(self.norm(x))

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
