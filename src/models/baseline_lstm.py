import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """
    Price-only LSTM baseline (Blueprint §2.2, Stage 0-1).
    Architecture: LSTM(input_dim -> 256) -> FC(256->128, ReLU, Dropout) -> FC(128->1)
    Loss: Huber (delta=0.01)
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        """x: (B, seq_len, features)"""
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]           # last hidden of top layer  (B, 256)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)

    # ── MC Dropout Inference (Blueprint §2.3) ────────────────────────────
    @torch.no_grad()
    def mc_predict(self, x, n_passes=50):
        """
        Monte-Carlo Dropout: 50 forward passes with dropout ON.
        Returns mean, conf_low (2.5%), conf_high (97.5%).
        """
        self.train()  # keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(n_passes)], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std
