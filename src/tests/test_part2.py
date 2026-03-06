"""
tests/smoke_test.py
Quick smoke test — verifies the full forward pass, loss, and MC Dropout
runs correctly on randomly generated mock data without real CSV files.

Run:
    cd part2
    python -m tests.smoke_test
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.tlstm import TLSTMModel
from src.training.trainer import Trainer, ewc_penalty
from src.utils.features import build_features


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_ohlcv(n_days=300, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(0, 5, n_days))
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    df = pd.DataFrame({
        "Open":   close * rng.uniform(0.99, 1.0, n_days),
        "High":   close * rng.uniform(1.00, 1.01, n_days),
        "Low":    close * rng.uniform(0.98, 0.99, n_days),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, n_days),
    }, index=dates)
    return df


def make_mock_loaders(batch_size=8):
    seq_len  = 30
    n_feat   = 9
    nlp_dim  = 514
    n_train  = 64
    n_val    = 16

    def make_batch(n):
        ohlcv = torch.randn(n, seq_len, n_feat)
        nlp   = torch.randn(n, nlp_dim)
        label = torch.randn(n, 1) * 0.02   # realistic pct move scale
        return TensorDataset(ohlcv, nlp, label)

    train_loader = DataLoader(make_batch(n_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(make_batch(n_val),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_feature_engineering():
    print("[1] Feature engineering ...", end=" ")
    df = make_mock_ohlcv()
    out = build_features(df)
    required = ["open_norm", "high_norm", "rsi_14", "ma_delta_5_20",
                "realised_vol_5", "regime_flag", "window_size"]
    for col in required:
        assert col in out.columns, f"Missing column: {col}"
    assert out["window_size"].isin([10, 30, 60]).all(), "Unexpected window size"
    print("OK")


def test_model_forward():
    print("[2] Model forward pass ...", end=" ")
    model = TLSTMModel()
    ohlcv = torch.randn(4, 60, 9)
    nlp   = torch.randn(4, 514)
    pred  = model(ohlcv, nlp)
    assert pred.shape == (4, 1), f"Expected (4,1), got {pred.shape}"
    print("OK")


def test_mc_dropout():
    print("[3] MC Dropout inference ...", end=" ")
    model = TLSTMModel()
    ohlcv = torch.randn(2, 30, 9)
    nlp   = torch.randn(2, 514)
    result = model.mc_dropout_predict(ohlcv, nlp, n_passes=10)
    for key in ["mean", "std", "ci_low", "ci_high"]:
        assert key in result and result[key].shape == (2, 1)
    assert (result["std"] >= 0).all(), "Std should be non-negative"
    print("OK")


def test_training_loop():
    print("[4] Training loop (2 epochs) ...", end=" ")
    device = torch.device("cpu")
    model  = TLSTMModel()
    trainer = Trainer(model, device, lr=1e-3, checkpoint_dir=__import__("pathlib").Path("/tmp/tlstm_smoke"))
    train_loader, val_loader = make_mock_loaders()

    import torch.optim as optim
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        trainer.train_epoch(train_loader, opt)

    metrics = trainer.evaluate(val_loader)
    assert "mae" in metrics and "direction_accuracy" in metrics
    assert 0.0 <= metrics["direction_accuracy"] <= 1.0
    print(f"OK  (val_mae={metrics['mae']:.4f}, dir_acc={metrics['direction_accuracy']:.2f})")


def test_ewc_penalty():
    print("[5] EWC penalty ...", end=" ")
    model = TLSTMModel()
    fisher = {
        name: torch.ones_like(p)
        for name, p in model.named_parameters()
        if "output_head" in name
    }
    old_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "output_head" in name
    }
    # Perturb model weights slightly
    with torch.no_grad():
        for p in model.output_head.parameters():
            p += 0.01
    penalty = ewc_penalty(model, fisher, old_params, ewc_lambda=400.0)
    assert penalty.item() > 0, "EWC penalty should be > 0 after perturbation"
    print(f"OK  (penalty={penalty.item():.4f})")


if __name__ == "__main__":
    print("\n=== T-LSTM Smoke Test ===\n")
    test_feature_engineering()
    test_model_forward()
    test_mc_dropout()
    test_training_loop()
    test_ewc_penalty()
    print("\n✓ All tests passed.\n")
