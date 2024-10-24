import torch.nn as nn
import torch.nn.functional as F
from .modules import ConformerBlock, LearnablePositionalEncoding


class Conformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ffn_dim, dropout):
        super(Conformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.input_projection(x)

        for block in self.conformer_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.final_projection(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class ConformerV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ffn_dim, dropout):
        super(ConformerV2, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = LearnablePositionalEncoding(hidden_dim)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for block in self.conformer_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.final_projection(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
