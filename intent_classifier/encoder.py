import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embed_dim
        )

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.feed_forward_norm = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.feed_forward_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Self Attention
        normalized_x = self.attention_norm(x)

        attention_output = self.attention(
            normalized_x,
            attention_mask,
        )

        attention_output = self.attention_dropout(
            attention_output
        )

        x = x + attention_output

        # Feed Forward
        normalized_x = self.feed_forward_norm(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        feed_forward_output = (
            self.feed_forward_dropout(
                feed_forward_output
            )
        )

        x = x + feed_forward_output

        return x
    
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int = 2,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(
                x,
                attention_mask,
            )

        return x