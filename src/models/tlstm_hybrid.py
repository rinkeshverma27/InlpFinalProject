"""
T-LSTM Dual-Stream Hybrid Model  (Blueprint §2.3)

Stream A: Pre-computed NLP embeddings (512-dim from FinBERT/MuRIL via handshake CSV)
Stream B: LSTM over OHLCV price features -> 256-dim hidden state
Fusion:   Concat -> 768-dim -> FC 128 (ReLU, Dropout) -> FC 1 (Linear)
Loss:     Huber (delta=0.01)
CI:       MC Dropout with 50 forward passes -> mean ± 1.96*std = 95% CI
"""

import torch
import torch.nn as nn


class TLSTMHybrid(nn.Module):
    def __init__(
        self,
        price_input_dim: int,
        nlp_embed_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Stream B: Price LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=price_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # ── Fusion head (768 = 512 NLP + 256 LSTM) ──────────────────────
        fusion_dim = nlp_embed_dim + lstm_hidden  # 768
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)   # kept active for MC Dropout
        self.fc2 = nn.Linear(128, 1)

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, price_seq, nlp_embed=None):
        """
        Args:
            price_seq: (B, seq_len, price_input_dim)
            nlp_embed: (B, 512) pre-computed trust-weighted NLP embedding.
                       If None, a zero vector is used (price-only fallback).
        """
        _, (h_n, _) = self.lstm(price_seq)
        lstm_out = h_n[-1]  # (B, 256)

        if nlp_embed is None:
            nlp_embed = torch.zeros(
                lstm_out.size(0), 512, device=lstm_out.device
            )

        fused = torch.cat([nlp_embed, lstm_out], dim=1)  # (B, 768)
        out = self.fc1(fused)
        out = self.relu(out)
        out = self.dropout(out)
        pred = self.fc2(out)
        return pred.squeeze(-1)

    # ── MC Dropout Inference (Blueprint §2.3) ────────────────────────────
    @torch.no_grad()
    def mc_predict(self, price_seq, nlp_embed=None, n_passes=50):
        """
        Monte-Carlo Dropout: run `n_passes` forward passes with dropout ON.
        Returns:
            mean_pred, conf_low, conf_high  (all tensors of shape (B,))
        """
        self.train()  # keep dropout active
        preds = torch.stack(
            [self.forward(price_seq, nlp_embed) for _ in range(n_passes)], dim=0
        )  # (n_passes, B)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std
