import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import deque


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings[:, :x.size(1), :]


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super(ConformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout1(attn_out))

        x_conv = x.transpose(1, 2)
        x_conv = self.dropout2(self.conv1(x_conv))
        x_conv = self.dropout2(self.conv2(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x = self.layer_norm2(x + x_conv)

        ffn_out = self.ffn(x)
        x = self.layer_norm3(x + self.dropout3(ffn_out))

        return x


class TripletLoss(nn.Module):
    def __init__(self, margin_min: float = 0.2, margin_max: float = 0.5, adjust_margin: bool = True,
                 hard_negative_ratio: float = 0.5):
        super().__init__()
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.current_margin = (margin_min + margin_max) / 2
        self.adjust_margin = adjust_margin
        self.margin_history = deque(maxlen=100)
        self.hard_negative_ratio = hard_negative_ratio

    def adjust_margin_value(self, loss_value: float):
        if not self.adjust_margin:
            return

        self.margin_history.append(loss_value)
        if len(self.margin_history) >= 50:
            avg_loss = np.mean(list(self.margin_history))
            if avg_loss < 0.1:
                self.current_margin = min(self.margin_max, self.current_margin + 0.02)
            elif avg_loss > 0.5:
                self.current_margin = max(self.margin_min, self.current_margin - 0.02)

    def cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(x1, x2)
        return 1 - cosine_sim

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        distance_positive = self.cosine_distance(anchor, positive)
        distance_negative = self.cosine_distance(anchor, negative)

        num_hard_negatives = int(self.hard_negative_ratio * len(distance_negative))
        hard_negatives_indices = torch.topk(distance_negative, num_hard_negatives, largest=True).indices
        hard_distance_negative = distance_negative[hard_negatives_indices]
        hard_distance_positive = distance_positive[hard_negatives_indices]

        losses = F.relu(hard_distance_positive - hard_distance_negative + self.current_margin)
        loss = losses.mean()

        self.adjust_margin_value(loss.item())

        return loss
