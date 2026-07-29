from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Configurable copy of the original LSTMClassifier2 architecture."""

    def __init__(
        self,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1,
        init_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_mode = init_mode

        self.embedding = nn.Linear(n_features, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.ReLU()
        self.hidden2out2 = nn.Linear(hidden_dim // 2, output_size)
        self.output = nn.Sigmoid()

    def _initial_state(self, batch_size: int, device: torch.device):
        shape = (self.num_layers, batch_size, self.hidden_dim)
        if self.init_mode == "random":
            h0 = torch.randn(shape, device=device)
            c0 = torch.randn(shape, device=device)
        elif self.init_mode == "zeros":
            h0 = torch.zeros(shape, device=device)
            c0 = torch.zeros(shape, device=device)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
        return h0, c0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        embeds = self.embedding(x)
        _, (hidden, _) = self.lstm(embeds, self._initial_state(batch_size, x.device))
        out = self.dropout_layer(hidden[-1])
        out = self.hidden2out(out)
        out = self.act(out)
        out = self.hidden2out2(out)
        return self.output(out)
