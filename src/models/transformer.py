import torch
from torch import nn


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        max_length=1024,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, max_length, hidden_size))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        length = x.size(1)
        z = self.input_projection(x) + self.position[:, :length]
        z = self.encoder(z)
        z = self.norm(z[:, -1])
        return self.head(z).squeeze(-1)
