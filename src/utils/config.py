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
