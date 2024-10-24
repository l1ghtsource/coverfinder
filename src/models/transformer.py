import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import PositionalEncoding, LearnablePositionalEncoding


class MusicTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int = 4096
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        self.pos_encoding = LearnablePositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.final_projection(x)

        x = F.normalize(x, p=2, dim=-1)
        return x
