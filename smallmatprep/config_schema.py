"""Configuration helpers."""
REQUIRED_KEYS = ["target"]


def validate_config(cfg: dict) -> list[str]:
    """Return a list of validation warnings/errors."""
    issues = []
    for k in REQUIRED_KEYS:
        if k not in cfg:
            issues.append(f"Missing required config key: {k}")
    return issues
