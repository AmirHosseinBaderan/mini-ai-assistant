import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.k_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.v_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_length, _ = x.shape

        x = x.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, _, seq_length, _ = x.shape

        x = x.transpose(1, 2)

        return x.contiguous().view(
            batch_size,
            seq_length,
            self.embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, seq_length, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embed_dim}, "
                f"got {embed_dim}"
            )
            
        # Q, K, V
        query = self.q_projection(x)
        key = self.k_projection(x)
        value = self.v_projection(x)

        # Split heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Attention scores
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self.head_dim
        )
        
        # Padding mask
        if attention_mask is not None:

            if attention_mask.shape != (
                batch_size,
                seq_length,
            ):
                raise ValueError(
                    "attention_mask must have shape "
                    "[batch_size, seq_length]"
                )

            mask = attention_mask[:, None, None, :]

            scores = scores.masked_fill(
                mask == 0,
                torch.finfo(scores.dtype).min,
            )

        # Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # Weighted values
        output = torch.matmul(
            attention_weights,
            value,
        )

        # Merge heads
        output = self._merge_heads(
            output
        )

        # Output projection
        output = self.output_projection(
            output
        )

        return output