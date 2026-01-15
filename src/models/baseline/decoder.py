import torch
import torch.nn as nn

class SpeciesDecoder(nn.Module):
    """
    Decoder for 342 plant species across Europe
    """
    def __init__(self, embed_dim, hidden_dim, output_dim=342, n_layers=1):
        super().__init__()

        # Build encoding layers
        layers = [
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            
        self.decoder = nn.Sequential(*layers)
        
        # Add final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Get logits for prediction
        """
        x = self.decoder(x)
        
        return self.output_layer(x)