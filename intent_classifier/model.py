import torch
from torch import nn

from .encoder import TransformerEncoder
from .positional_encoding import PositionalEncoding


class IntentTransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_length: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.position = PositionalEncoding(
            embed_dim=embed_dim,
            max_length=max_length,
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.embedding(input_ids)

        x = self.position(x)

        x = self.dropout(x)

        x = self.encoder(
            x,
            attention_mask,
        )

        return x