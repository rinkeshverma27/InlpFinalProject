"""
src/utils/config_loader.py — Load config.yaml and apply active VRAM profile.
Import this everywhere: from src.utils.config_loader import load_config
"""

import yaml
import pathlib
from src.utils.logger import get_logger

log = get_logger("config")

_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: pathlib.Path = _CONFIG_PATH, profile_override: str = None) -> dict:
    """
    Load config.yaml and overlay the active VRAM profile's values onto the
    model/training sections. Returns a single merged dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile_name = profile_override or cfg.get("vram_profile", "8gb")
    profiles     = cfg.get("profiles", {})
    profile      = profiles.get(profile_name, {})

    if not profile:
        log.warning(f"[CONFIG] Unknown vram_profile '{profile_name}'. Using defaults.")

    # Apply profile overrides
    model    = cfg.setdefault("model", {})
    training = cfg.setdefault("training", {})
    nlp      = cfg.setdefault("nlp", {})

    mapping = {
        "price_hidden_size":     ("model",    "price_hidden_size"),
        "price_num_layers":      ("model",    "price_num_layers"),
        "sentiment_hidden_size": ("model",    "sentiment_hidden_size"),
        "sentiment_num_layers":  ("model",    "sentiment_num_layers"),
        "fc_hidden_dim":         ("model",    "fc_hidden_dim"),
        "mc_dropout_passes":     ("model",    "mc_dropout_passes"),
        "use_int8_inference":    ("model",    "use_int8_inference"),
        "finbert_batch_size":    ("nlp",      "finbert_batch_size"),
        "muril_batch_size":      ("nlp",      "muril_batch_size"),
        "train_batch_size":      ("training", "batch_size"),
        "muril_train_batch_size":("training", "muril_batch_size"),
        "epochs":                ("training", "epochs"),
        "muril_epochs":          ("training", "muril_epochs"),
    }

    for prof_key, (section, cfg_key) in mapping.items():
        if prof_key in profile:
            cfg[section][cfg_key] = profile[prof_key]

    # Auto-compute fusion_dim
    model["fusion_dim"] = (
        model.get("price_hidden_size", 192) +
        model.get("sentiment_hidden_size", 64)
    )

    log.info(
        f"[CONFIG] Profile='{profile_name}' | "
        f"price_hidden={model['price_hidden_size']} | "
        f"sent_hidden={model['sentiment_hidden_size']} | "
        f"epochs={training['epochs']} | "
        f"int8={model.get('use_int8_inference', False)}"
    )
    return cfg
