import torch
import torch.nn as nn

def get_layer(in_d, out_d, lip=False):
    if lip: return nn.utils.spectral_norm(nn.Linear(in_d, out_d))
    else: return nn.Linear(in_d, out_d)

class MLP(nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1,
                 leaky=0.1, factor=64, n_layers=2, lip=False,dropout=0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.non_linear = nn.LeakyReLU(leaky) if leaky > 0 else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # First layer
        if dropout==0:
            self.layers = nn.ModuleList([get_layer(dim, factor, lip=lip)])
            for _ in range(n_layers):
                self.layers.append(get_layer(factor, factor, lip=lip))
        else:
            self.layers = nn.ModuleList([self.dropout(get_layer(dim, factor, lip=lip))])
            for _ in range(n_layers):
                self.layers.append(self.dropout(get_layer(factor, factor, lip=lip)))

        # Last layer
        self.layers.append(get_layer(factor, z_dim,lip=lip))

        self.to(device)
        self.device = device

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # Apply non-linearity on all but last layer
                h = self.non_linear(h)
        return h

