"""AI-enhanced utilities for SmallMatPrep (optional)."""
from .client import LLMClient
from .config_generator import generate_config, generate_config_with_columns, save_config
from .diagnose_reporter import diagnose

__all__ = [
    "generate_config",
    "generate_config_with_columns",
    "save_config",
    "diagnose",
    "LLMClient",
]
