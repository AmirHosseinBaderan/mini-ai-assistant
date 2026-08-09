import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        max_length: int = 64,
    ):
        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_dim,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / embed_dim)
        )

        encoding = torch.zeros(
            max_length,
            embed_dim,
        )

        encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        encoding[:, 1::2] = torch.cos(
            position * div_term
        )

        encoding = encoding.unsqueeze(0)

        self.register_buffer(
            "encoding",
            encoding,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence_length = x.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError(
                "Sequence length exceeds max_length"
            )

        return x + self.encoding[
            :, :sequence_length, :
        ]