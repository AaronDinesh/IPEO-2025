import torch
import torch.nn as nn

class ClimateVarsEncoder(nn.Module):
    """
    Encoder for climatic variables
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, layers=2):
        super().__init__()
        # Create hidden layers for MLP
        layers_list = []
        in_dim = input_dim
        for _ in range(layers):
            layers_list.append(nn.Linear(in_dim, hidden_dim))
            layers_list.append(nn.GELU())
            in_dim = hidden_dim
        layers_list.append(nn.Linear(hidden_dim, embed_dim))
        layers_list.append(nn.LayerNorm(embed_dim))
        
        self.mlp = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.mlp(x)

class TimeSeriesEncoder(nn.Module):
    """
    Encoder for RGB + NIR time series data
    """
    def __init__(self, embed_dim, layers, dropout):
        super().__init__()
        #self.rnn = nn.GRU(4, embed_dim, num_layers=layers, batch_first=True, dropout=dropout)
        self.lstm = nn.LSTM(4, embed_dim, num_layers=layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        
        return self.norm(output[:, -1])