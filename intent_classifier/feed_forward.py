import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(
            embed_dim,
            ff_dim,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(
            ff_dim,
            embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear_1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.linear_2(x)

        return x