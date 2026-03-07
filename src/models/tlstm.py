"""
models/tlstm.py
Transformer-LSTM Hybrid (T-LSTM) Predictive Model.

Architecture:
  Stream B → LSTM Encoder → 256-dim hidden state ──┐
                                                     ├─ Concat (768-dim) → FC Head → 1 float
  Stream A → NLP Vector (514-dim compressed) ───────┘

Note: Stream A embeddings (512-dim) are condensed to 256-dim via a projection
      layer before concatenation, making fusion 512-dim total.
      The 768-dim design from the blueprint is achieved by projecting NLP 514→512
      then concat with 256-dim LSTM output = 768-dim total.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Encoder (Stream B)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Processes a padded sequence of OHLCV features and returns
    the last-timestep hidden state.

    Input  : (batch, seq_len, n_features)
    Output : (batch, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden_dim) → take the last layer
        return h_n[-1]   # (B, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# NLP Projection (Stream A compressor)
# ──────────────────────────────────────────────────────────────────────────────

class NLPProjection(nn.Module):
    """
    Projects the 514-dim NLP vector (512 trust-weighted embedding + 2 sentiments)
    to 512-dim to match the blueprint's 512 Transformer embedding dimension.
    """

    def __init__(self, in_dim: int = 514, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, 512)


# ──────────────────────────────────────────────────────────────────────────────
# Output Head (FC Layers)
# ──────────────────────────────────────────────────────────────────────────────

class OutputHead(nn.Module):
    """
    Dual-Head Output:
    1. Direction (Classification): Logits for Up/Down movement.
    2. Magnitude (Regression): Predicted adjusted % move.
    """

    def __init__(self, fusion_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.direction = nn.Linear(128, 1)  # Logits
        self.magnitude = nn.Linear(128, 1)  # Raw float

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.common(x)
        return {
            "direction": self.direction(feat),
            "magnitude": self.magnitude(feat),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Full T-LSTM Model
# ──────────────────────────────────────────────────────────────────────────────

class TLSTMModel(nn.Module):
    """
    Full Transformer-LSTM Hybrid model.

    Inputs
    ------
    ohlcv_seq : (B, T, n_ohlcv_features)   — windowed OHLCV features
    nlp_vec   : (B, 514)                    — trust-weighted NLP embedding

    Output
    ------
    dict:
        direction : (B, 1) - logits for classification
        magnitude : (B, 1) - raw float for regression

    Inference
    ---------
    Call mc_dropout_predict() for mean + 95% CI.
    """

    def __init__(
        self,
        n_ohlcv_features: int = 11,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        nlp_in_dim: int = 514,
        nlp_out_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm_encoder   = LSTMEncoder(n_ohlcv_features, lstm_hidden_dim, lstm_layers, lstm_dropout)
        self.nlp_projection = NLPProjection(nlp_in_dim, nlp_out_dim)
        self.output_head    = OutputHead(lstm_hidden_dim + nlp_out_dim, head_dropout)

    def forward(self, ohlcv_seq: torch.Tensor, nlp_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        h_lstm = self.lstm_encoder(ohlcv_seq)       # (B, 256)
        h_nlp  = self.nlp_projection(nlp_vec)       # (B, 512)
        fused  = torch.cat([h_nlp, h_lstm], dim=-1) # (B, 768)
        return self.output_head(fused)

    @torch.no_grad()
    def mc_dropout_predict(
        self,
        ohlcv_seq: torch.Tensor,
        nlp_vec: torch.Tensor,
        n_passes: int = 50,
    ) -> dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout inference.
        """
        self.train()   # enable dropout layers
        outputs = [self.forward(ohlcv_seq, nlp_vec) for _ in range(n_passes)]
        
        dir_preds = torch.stack([o["direction"] for o in outputs], dim=0) # (n_passes, B, 1)
        mag_preds = torch.stack([o["magnitude"] for o in outputs], dim=0) # (n_passes, B, 1)
        
        self.eval()

        # For direction, we average the logits or probabilities. 
        # Here we'll average the sigmoid outputs for a "probability" mean.
        dir_probs = torch.sigmoid(dir_preds).mean(dim=0)
        
        mean_mag = mag_preds.mean(dim=0)
        std_mag  = mag_preds.std(dim=0)
        
        return {
            "direction_prob": dir_probs,
            "mean":           mean_mag,
            "std":            std_mag,
            "ci_low":         mean_mag - 1.96 * std_mag,
            "ci_high":        mean_mag + 1.96 * std_mag,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Fisher Information Matrix for EWC (Tier 2 weekly nudge)
# ──────────────────────────────────────────────────────────────────────────────

def compute_fisher_matrix(
    model: TLSTMModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """
    Approximate the diagonal Fisher Information Matrix for EWC.
    """
    model.train()
    fisher = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
        if "output_head" in name and p.requires_grad
    }

    count = 0
    # Use the same dual loss logic as the trainer for consistency
    bce_fn = nn.BCEWithLogitsLoss(reduction="sum")
    huber_fn = nn.HuberLoss(delta=0.01, reduction="sum")

    for ohlcv_seq, nlp_vec, label in dataloader:
        if count >= n_samples:
            break
        ohlcv_seq, nlp_vec, label = (
            ohlcv_seq.to(device), nlp_vec.to(device), label.to(device)
        )
        model.zero_grad()
        outputs = model(ohlcv_seq, nlp_vec)
        
        # Combined Loss derivation for Fisher
        dir_target = (label > 0).float()
        loss_dir = bce_fn(outputs["direction"], dir_target)
        loss_mag = huber_fn(outputs["magnitude"], label)
        
        loss = 0.6 * loss_dir + 0.4 * loss_mag
        loss.backward()

        for name, p in model.named_parameters():
            if name in fisher and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2

        count += ohlcv_seq.size(0)

    for name in fisher:
        fisher[name] /= count

    model.eval()
    return fisher
