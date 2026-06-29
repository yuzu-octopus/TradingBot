import warnings

import torch
from torch import nn

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


class MarketGate(nn.Module):
    def __init__(
        self, n_features: int, market_state_size: int = 5, hidden: int = 16
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(market_state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(
        self, stock_features: torch.Tensor, market_state: torch.Tensor
    ) -> torch.Tensor:
        gate = self.encoder(market_state).unsqueeze(1)
        return stock_features * gate


class RankGLU(nn.Module):
    def __init__(
        self, d_model: int, bottleneck: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.shortcut = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, bottleneck)
        self.gate = nn.Linear(d_model, bottleneck)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(bottleneck, 1)
        self.gamma = nn.Parameter(
            torch.ones(1)
        )  # init at 1.0 so both branches contribute equally

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        shortcut = self.shortcut(x)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        nonlinear = self.out(self.dropout(v * g))
        return shortcut + self.gamma * nonlinear


class StockTransformer(nn.Module):
    def __init__(
        self,
        n_stocks: int,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        rankglu_bottleneck: int = 64,
        market_state_size: int = 0,
    ) -> None:
        super().__init__()
        self.n_stocks = n_stocks
        self.market_state_size = market_state_size
        self.input_proj = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        if market_state_size > 0:
            self.market_gate = MarketGate(n_features, market_state_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_head = RankGLU(
            d_model, bottleneck=rankglu_bottleneck, dropout=dropout
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, market_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        if market_state is not None and self.market_state_size > 0:
            x = self.market_gate(x, market_state)
        x = self.input_proj(x)
        x = self.dropout(x)
        # Encoder-only with full bidirectional self-attention — every stock
        # attends to every stock. No causal mask, no permutation, no
        # stock_embed (full attention captures cross-stock relationships
        # without needing positional identity embeddings).
        x = self.transformer(x)
        x = self.norm(x)
        x = self.output_head(x)
        # Clamp instead of tanh: bounds scores to [-1, 1] without the
        # gradient vanishing that tanh causes near saturation. Only kills
        # gradients when a value is exactly outside [-1, 1], not gradually.
        x = torch.clamp(x, -1, 1)
        return x.squeeze(-1)
