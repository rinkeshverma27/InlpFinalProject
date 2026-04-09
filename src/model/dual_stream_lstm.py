"""
src/model/dual_stream_lstm.py — True dual-stream LSTM for stock direction prediction.

Stream B (Price):     3-layer unidirectional LSTM, hidden=192 (8GB profile)
Stream A (Sentiment): 2-layer unidirectional LSTM, hidden=64
Fusion:               concat → LayerNorm → Dropout → FC(256→128) → GELU → FC(128→1) → Sigmoid

All dims are read from cfg, never hardcoded.
Variational dropout: same mask applied at every timestep (theoretically correct for RNNs).
"""

import torch
import torch.nn as nn
from typing import Tuple


class VariationalDropout(nn.Module):
    """Apply the same dropout mask at every timestep in a sequence."""

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [B, T, H] or [B, H]
        if not self.training or self.p == 0:
            return x
        if x.dim() == 3:
            mask = x.new_ones(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        else:
            mask = x.new_ones(x.size(0), x.size(1)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class LSTMStream(nn.Module):
    """Single LSTM branch with variational dropout between layers."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers  = nn.ModuleList()
        self.dropouts= nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.LSTM(in_dim, hidden_dim, batch_first=True))
            self.dropouts.append(VariationalDropout(dropout))
            in_dim = hidden_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim] → returns [B, hidden_dim]
        out = x
        for i, (lstm, drop) in enumerate(zip(self.layers, self.dropouts)):
            out, _ = lstm(out)   # [B, T, H]
            if i < len(self.layers) - 1:
                out = drop(out)  # apply to full sequence for intermediate layers
            else:
                # Last layer: only dropout the final timestep before returning
                out = out[:, -1, :]
                out = drop(out)
                return out
        return out[:, -1, :]


class DualStreamLSTM(nn.Module):
    """
    Dual-stream LSTM:
      - Price branch   (Stream B): deeper, 3 layers
      - Sentiment branch (Stream A): lighter, 2 layers
      - Fusion + prediction head
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        price_in   = m.get("price_input_dim",       11)
        price_h    = m.get("price_hidden_size",      192)
        price_l    = m.get("price_num_layers",       3)
        price_drop = m.get("price_dropout",          0.3)

        sent_in    = m.get("sentiment_input_dim",    9)
        sent_h     = m.get("sentiment_hidden_size",  64)
        sent_l     = m.get("sentiment_num_layers",   2)
        sent_drop  = m.get("sentiment_dropout",      0.3)

        fusion_dim = m.get("fusion_dim") or (price_h + sent_h)
        fc_hidden  = m.get("fc_hidden_dim",          128)
        out_drop   = m.get("output_dropout",         0.3)

        self.price_stream  = LSTMStream(price_in, price_h, price_l, price_drop)
        self.sent_stream   = LSTMStream(sent_in,  sent_h,  sent_l,  sent_drop)
        self.norm          = nn.LayerNorm(fusion_dim)
        self.out_dropout   = nn.Dropout(out_drop)
        self.fc1           = nn.Linear(fusion_dim, fc_hidden)
        self.act           = nn.GELU()
        self.fc2           = nn.Linear(fc_hidden, 1)

    def forward(
        self,
        price_seq: torch.Tensor,      # [B, T, 11]
        sentiment_seq: torch.Tensor,  # [B, T, 9]
    ) -> torch.Tensor:
        h_price = self.price_stream(price_seq)        # [B, price_h]
        h_sent  = self.sent_stream(sentiment_seq)     # [B, sent_h]

        fused   = torch.cat([h_price, h_sent], dim=-1)  # [B, fusion_dim]
        fused   = self.norm(fused)
        fused   = self.out_dropout(fused)
        out     = self.fc1(fused)
        out     = self.act(out)
        out     = self.fc2(out)                       # [B, 1]  raw logit
        return out.squeeze(-1)                        # [B]     Logits

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> DualStreamLSTM:
    model = DualStreamLSTM(cfg)
    from src.utils.logger import get_logger
    log = get_logger("model")
    log.info(
        f"DualStreamLSTM built | "
        f"price_hidden={cfg['model'].get('price_hidden_size', 192)} "
        f"×{cfg['model'].get('price_num_layers', 3)}L | "
        f"sent_hidden={cfg['model'].get('sentiment_hidden_size', 64)} "
        f"×{cfg['model'].get('sentiment_num_layers', 2)}L | "
        f"params={model.n_params:,}"
    )
    return model
