import yaml
from pathlib import Path
from .paths import CONFIG_FILE

def load_config(config_path=CONFIG_FILE):
    """Loads the main project configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

# Load config statically so it can be imported directly
global_config = load_config()
from pathlib import Path
import os

# Base directory is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NEWS_RAW_DIR = DATA_DIR / "news" / "raw"
NEWS_PROCESSED_DIR = DATA_DIR / "news" / "processed"
HOLDOUT_DATA_DIR = DATA_DIR / "holdout"
PREDICTIONS_DIR = DATA_DIR / "predictions"
TRUST_WEIGHTS_DIR = DATA_DIR / "trust_weights"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, NEWS_RAW_DIR, NEWS_PROCESSED_DIR, HOLDOUT_DATA_DIR, PREDICTIONS_DIR, TRUST_WEIGHTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Important Files
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yml"
HANDSHAKE_CSV = PROCESSED_DATA_DIR / "handshake.csv"
PREDICTIONS_CSV = PREDICTIONS_DIR / "predictions.csv"
