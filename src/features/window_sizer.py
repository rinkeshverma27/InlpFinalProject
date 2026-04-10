"""
src/features/window_sizer.py — Dynamic sequence window based on volatility regime.

In high-volatility regimes the market changes fast; old context goes stale quickly.
In calm regimes a longer lookback captures useful trend information.
"""


def get_window(atr_percentile: float, w_min: int = 10, w_max: int = 60, scale: float = 50) -> int:
    """
    Compute the LSTM lookback window length for a given ATR regime.

    Args:
        atr_percentile : Rolling percentile rank of ATR(14)/close for this day [0, 1].
        w_min          : Floor window length (high volatility).
        w_max          : Ceiling window length (low volatility).
        scale          : Sensitivity multiplier.

    Returns:
        Integer window length in [w_min, w_max].

    Design:
        window = max(w_min, w_max - int(atr_percentile * scale))
        ATR p=0.9 → window = max(10, 60 - 45) = 15   (fast market)
        ATR p=0.1 → window = max(10, 60 -  5) = 55   (calm trend)
    """
    return max(w_min, w_max - int(atr_percentile * scale))
