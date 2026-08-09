import torch
from torch import nn


class MeanPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        masked_hidden_states = (
            hidden_states * mask
        )

        token_count = mask.sum(
            dim=1
        ).clamp_min(1.0)

        pooled = (
            masked_hidden_states.sum(dim=1)
            / token_count
        )

        return pooled