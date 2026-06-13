import os
import yaml
from pathlib import Path

def load_config(config_path: str = "config/phase1_config.yaml") -> dict:
    """Load YAML config, return dict. Print error and exit if not found."""
    full_path = Path(config_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path.absolute()}")
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty or malformed: {full_path}")
    return config